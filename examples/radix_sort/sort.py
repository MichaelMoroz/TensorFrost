from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

import numpy as np

import TensorFrost as tf

__all__ = ["HistogramRadixSort", "radix_sort"]

_TYPE_CODES: Dict[str, np.uint32] = {
	"uint": np.uint32(0),
	"int": np.uint32(1),
	"float": np.uint32(2),
}


def _dispatch_groups(work_items: int, threads_per_group: int) -> int:
	if work_items <= 0:
		return 0
	return (work_items + threads_per_group - 1) // threads_per_group


def _prepare_keys(keys: np.ndarray) -> Tuple[np.ndarray, np.dtype, str]:
	array = np.asarray(keys)
	if array.ndim != 1:
		raise ValueError("radix_sort expects a 1D array of keys")

	dtype = array.dtype
	if dtype == np.uint32:
		return array, dtype, "uint"

	if dtype == np.int32:
		return array, dtype, "int"

	if dtype == np.float32:
		return array, dtype, "float"

	raise TypeError(f"Unsupported key dtype {dtype}; expected uint32, int32, or float32")


def _prepare_values(values: np.ndarray) -> Tuple[np.ndarray, np.dtype]:
	array = np.asarray(values)
	if array.ndim != 1:
		raise ValueError("radix_sort expects a 1D array of values when provided")

	dtype = array.dtype
	if dtype not in (np.uint32, np.int32, np.float32):
		raise TypeError(f"Unsupported value dtype {dtype}; expected uint32, int32, or float32")

	return array, dtype


_SHADER_DIR = Path(__file__).resolve().parent / "shaders"


def _load_shader_source(filename: str) -> str:
	shader_path = _SHADER_DIR / filename
	return shader_path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class _SorterKey:
	bits_per_pass: int
	block_size: int
	group_size: int


class HistogramRadixSort:
	"""GPU histogram radix sort implemented with Slang + Vulkan."""

	def __init__(self, *, bits_per_pass: int = 6, block_size: int = 64, group_size: int = 128) -> None:
		if bits_per_pass <= 0:
			raise ValueError("bits_per_pass must be positive")
		if bits_per_pass > 8:
			raise ValueError("bits_per_pass must be <= 8 to fit within MAX_HIST_SIZE")
		if group_size != 128:
			raise ValueError("This implementation currently requires group_size == 128")
		if block_size <= 0 or block_size > 1024:
			raise ValueError("block_size must be within (0, 1024]")

		self.bits_per_pass = bits_per_pass
		self.block_size = block_size
		self.group_size = group_size
		self.histogram_size = 1 << bits_per_pass
		self.last_stage_timings = None
		self.last_validation_errors: Optional[int] = None

		def inject_defines(filename: str, *, with_group: bool = False, with_histogram: bool = False) -> str:
			defines = []
			if with_group:
				defines.append(f"#define TF_GROUP_SIZE {self.group_size}u")
			if with_histogram:
				defines.append(f"#define TF_HISTOGRAM_SIZE {self.histogram_size}u")
			source = _load_shader_source(filename)
			if defines:
				return "\n".join(defines) + "\n" + source
			return source

		self._map_to_uint_program = tf.createComputeProgramFromSlang(
			"radix_map_to_uint",
			inject_defines("map_to_uint.slang", with_group=True),
			"csMapToUint",
			ro_count=2,
			rw_count=1,
		)
		self._map_from_uint_program = tf.createComputeProgramFromSlang(
			"radix_map_from_uint",
			inject_defines("map_from_uint.slang", with_group=True),
			"csMapFromUint",
			ro_count=2,
			rw_count=1,
		)

		self._histogram_program = tf.createComputeProgramFromSlang(
			"radix_histogram",
			inject_defines("histogram.slang", with_group=True),
			"csHistogram",
			ro_count=2,
			rw_count=1,
		)
		self._unpack_program = tf.createComputeProgramFromSlang(
			"radix_unpack",
			_load_shader_source("unpack.slang"),
			"csUnpack",
			ro_count=2,
			rw_count=1,
		)
		self._prefix_local_program = tf.createComputeProgramFromSlang(
			"radix_prefix_local",
			_load_shader_source("prefix_local.slang"),
			"csPrefixLocal",
			ro_count=2,
			rw_count=2,
		)
		self._prefix_blocks_program = tf.createComputeProgramFromSlang(
			"radix_prefix_blocks",
			_load_shader_source("prefix_block.slang"),
			"csPrefixBlocks",
			ro_count=2,
			rw_count=1,
		)
		self._prefix_accum_program = tf.createComputeProgramFromSlang(
			"radix_prefix_accum",
			_load_shader_source("prefix_accum.slang"),
			"csPrefixAccumulate",
			ro_count=2,
			rw_count=1,
		)
		self._bucket_scan_program = tf.createComputeProgramFromSlang(
			"radix_bucket_scan",
			_load_shader_source("bucket_scan.slang"),
			"csBucketScan",
			ro_count=2,
			rw_count=1,
		)
		scatter_source = inject_defines("scatter.slang", with_group=True, with_histogram=True)
		self._scatter_program = tf.createComputeProgramFromSlang(
			"radix_scatter",
			scatter_source,
			"csScatter",
			ro_count=5,
			rw_count=2,
		)

		# Validation program: checks if adjacent key pairs are sorted (after mapping back to original type)
		self._validate_program = tf.createComputeProgramFromSlang(
			"radix_validate_sorted",
			inject_defines("validate_sorted.slang", with_group=True),
			"csValidate",
			ro_count=2,
			rw_count=1,
		)

		self._dummy_values_buffer = tf.createBuffer(1, 4, False)

	def close(self) -> None:
		for program in (
			self._map_to_uint_program,
			self._map_from_uint_program,
			self._histogram_program,
			self._unpack_program,
			self._prefix_local_program,
			self._prefix_blocks_program,
			self._prefix_accum_program,
			self._bucket_scan_program,
			self._scatter_program,
			self._validate_program,
		):
			if program is not None:
				program.release()
		if self._dummy_values_buffer is not None:
			self._dummy_values_buffer.release()

	def sort(
		self,
		keys: np.ndarray,
		values: Optional[np.ndarray] = None,
		*,
		max_bits: int = 32,
		collect_stage_timings: bool = False,
		validate: bool = False,
		return_arrays: bool = True,
	) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		keys_array, key_dtype, key_kind = _prepare_keys(keys)
		element_count = int(keys_array.shape[0])

		if values is not None:
			values_array, values_dtype = _prepare_values(values)
			if values_array.shape[0] != element_count:
				raise ValueError("values must have the same length as keys")
		else:
			values_array = None
			values_dtype = None

		if element_count == 0:
			empty_keys = keys_array.copy()
			self.last_stage_timings = {} if collect_stage_timings else None
			if validate:
				self.last_validation_errors = 0
			if values_array is None:
				return empty_keys, None
			return empty_keys, values_array.copy()

		max_bits = int(min(max_bits, 32))
		histogram_size = self.histogram_size
		mask = np.uint32(histogram_size - 1)

		num_groups = max((element_count + self.group_size - 1) // self.group_size, 1)
		block_count = max((num_groups + self.block_size - 1) // self.block_size, 1)
		packed_count = (histogram_size + 3) // 4
		passes = max((max_bits + self.bits_per_pass - 1) // self.bits_per_pass, 1)

		params_array = np.zeros(8, dtype=np.uint32)
		params_array[0] = np.uint32(element_count)
		params_array[1] = np.uint32(histogram_size)
		params_array[3] = mask
		params_array[4] = np.uint32(num_groups)
		params_array[5] = np.uint32(self.block_size)
		params_array[6] = np.uint32(block_count)
		params_array[7] = np.uint32(1 if values_array is not None else 0)

		map_params = np.zeros(4, dtype=np.uint32)
		map_params[0] = np.uint32(element_count)
		map_params[1] = _TYPE_CODES[key_kind]

		if collect_stage_timings:
			stage_totals = {
				"map_to_uint": 0.0,
				"histogram": 0.0,
				"unpack": 0.0,
				"prefix_local": 0.0,
				"prefix_blocks": 0.0,
				"prefix_accum": 0.0,
				"bucket_scan": 0.0,
				"scatter": 0.0,
				"map_from_uint": 0.0,
			}
		else:
			stage_totals = {}

		params_buffer = tf.createBuffer(params_array.size, 4, True)
		params_buffer.setData(params_array)

		map_buffer = tf.createBuffer(map_params.size, 4, True)
		map_buffer.setData(map_params)

		key_buffers = [tf.createBuffer(max(element_count, 1), 4, False) for _ in range(2)]
		key_buffers[0].setData(keys_array)

		if values_array is not None:
			value_buffers = [tf.createBuffer(max(element_count, 1), 4, False) for _ in range(2)]
			value_buffers[0].setData(values_array)
		else:
			dummy = self._dummy_values_buffer
			value_buffers = [dummy, dummy]

		packed_hist_buffer = tf.createBuffer(max(packed_count * num_groups, 1), 4, False)
		group_hist_buffer = tf.createBuffer(max(histogram_size * num_groups, 1), 4, False)
		prefix_buffer = tf.createBuffer(max(histogram_size * num_groups, 1), 4, False)
		block_totals_buffer = tf.createBuffer(max(histogram_size * block_count, 1), 4, False)
		block_prefix_buffer = tf.createBuffer(max(histogram_size * block_count, 1), 4, False)
		bucket_scan_buffer = tf.createBuffer(max(histogram_size, 1), 4, False)

		map_groups = _dispatch_groups(element_count, self.group_size)
		reduction_group_size = 64
		unpack_groups = _dispatch_groups(histogram_size * num_groups, reduction_group_size)
		prefix_local_groups = _dispatch_groups(histogram_size * block_count, reduction_group_size)
		prefix_block_groups = _dispatch_groups(histogram_size, reduction_group_size)
		prefix_accum_groups = _dispatch_groups(histogram_size * block_count, reduction_group_size)
		bucket_scan_groups = _dispatch_groups(histogram_size, reduction_group_size)
		scatter_groups = num_groups
		histogram_groups = num_groups

		# Total pass timer starts at the first dispatch and ends after the last (map_from_uint)
		total_start = time.perf_counter() if collect_stage_timings else None
		start = time.perf_counter() if collect_stage_timings else None
		self._map_to_uint_program.run(
			[map_buffer, key_buffers[0]],
			[key_buffers[1]],
			map_groups,
		)
		if collect_stage_timings and start is not None:
			stage_totals["map_to_uint"] += time.perf_counter() - start

		key_in = key_buffers[1]
		key_out = key_buffers[0]
		val_in, val_out = value_buffers

		for pass_index in range(passes):
			params_array[2] = np.uint32(pass_index * self.bits_per_pass)
			params_buffer.setData(params_array)

			start = time.perf_counter() if collect_stage_timings else None
			self._histogram_program.run(
				[params_buffer, key_in],
				[packed_hist_buffer],
				histogram_groups,
			)
			if collect_stage_timings and start is not None:
				stage_totals["histogram"] += time.perf_counter() - start

			start = time.perf_counter() if collect_stage_timings else None
			self._unpack_program.run(
				[params_buffer, packed_hist_buffer],
				[group_hist_buffer],
				unpack_groups,
			)
			if collect_stage_timings and start is not None:
				stage_totals["unpack"] += time.perf_counter() - start

			start = time.perf_counter() if collect_stage_timings else None
			self._prefix_local_program.run(
				[params_buffer, group_hist_buffer],
				[prefix_buffer, block_totals_buffer],
				prefix_local_groups,
			)
			if collect_stage_timings and start is not None:
				stage_totals["prefix_local"] += time.perf_counter() - start

			start = time.perf_counter() if collect_stage_timings else None
			self._prefix_blocks_program.run(
				[params_buffer, block_totals_buffer],
				[block_prefix_buffer],
				prefix_block_groups,
			)
			if collect_stage_timings and start is not None:
				stage_totals["prefix_blocks"] += time.perf_counter() - start

			start = time.perf_counter() if collect_stage_timings else None
			self._prefix_accum_program.run(
				[params_buffer, block_prefix_buffer],
				[prefix_buffer],
				prefix_accum_groups,
			)
			if collect_stage_timings and start is not None:
				stage_totals["prefix_accum"] += time.perf_counter() - start

			start = time.perf_counter() if collect_stage_timings else None
			self._bucket_scan_program.run(
				[params_buffer, prefix_buffer],
				[bucket_scan_buffer],
				bucket_scan_groups,
			)
			if collect_stage_timings and start is not None:
				stage_totals["bucket_scan"] += time.perf_counter() - start

			start = time.perf_counter() if collect_stage_timings else None
			self._scatter_program.run(
				[params_buffer, key_in, val_in, prefix_buffer, bucket_scan_buffer],
				[key_out, val_out],
				scatter_groups,
			)
			if collect_stage_timings and start is not None:
				stage_totals["scatter"] += time.perf_counter() - start

			key_in, key_out = key_out, key_in
			if values_array is not None:
				val_in, val_out = val_out, val_in

		start = time.perf_counter() if collect_stage_timings else None
		self._map_from_uint_program.run(
			[map_buffer, key_in],
			[key_out],
			map_groups,
		)
		if collect_stage_timings and start is not None:
			stage_totals["map_from_uint"] += time.perf_counter() - start

		# Record total time from first to last dispatch in the sort pass
		if collect_stage_timings and total_start is not None:
			stage_totals["total_pass"] = time.perf_counter() - total_start

		# Optional GPU-side validation: check adjacent pairs and atomically count violations
		if validate:
			validate_params = np.zeros(2, dtype=np.uint32)
			validate_params[0] = np.uint32(element_count)
			validate_params[1] = map_params[1]  # type code

			validate_params_buf = tf.createBuffer(validate_params.size, 4, True)
			validate_params_buf.setData(validate_params)

			error_buf = tf.createBuffer(1, 4, False)
			error_zero = np.zeros(1, dtype=np.uint32)
			error_buf.setData(error_zero)

			# Reuse map_groups; kernel early-outs for i >= n-1
			self._validate_program.run(
				[validate_params_buf, key_out],
				[error_buf],
				map_groups,
			)

			error_count = int(error_buf.getData(np.dtype(np.uint32), 1)[0])
			self.last_validation_errors = error_count

		if return_arrays:
			sorted_keys = key_out.getData(key_dtype, element_count)
			if values_array is not None and values_dtype is not None:
				sorted_values = val_in.getData(values_dtype, element_count)
			else:
				sorted_values = None
		else:
			# Avoid full readback when not needed by caller
			sorted_keys = np.empty(0, dtype=key_dtype)
			sorted_values = (np.empty(0, dtype=values_dtype) if values_array is not None and values_dtype is not None else None)

		self.last_stage_timings = stage_totals if collect_stage_timings else None
		return sorted_keys, sorted_values


_SORTER_CACHE: Dict[_SorterKey, HistogramRadixSort] = {}


def _get_sorter(bits_per_pass: int, block_size: int, group_size: int) -> HistogramRadixSort:
	key = _SorterKey(bits_per_pass, block_size, group_size)
	sorter = _SORTER_CACHE.get(key)
	if sorter is None:
		sorter = HistogramRadixSort(bits_per_pass=bits_per_pass, block_size=block_size, group_size=group_size)
		_SORTER_CACHE[key] = sorter
	return sorter


def radix_sort(
	keys: np.ndarray,
	values: Optional[np.ndarray] = None,
	*,
	bits_per_pass: int = 6,
	max_bits: int = 32,
	block_size: int = 64,
	group_size: int = 128,
):
	"""Run the GPU histogram radix sort on the provided keys (and optional values).

	Returns the sorted keys, and when ``values`` is provided also returns the permuted values.
	"""

	sorter = _get_sorter(bits_per_pass, block_size, group_size)
	sorted_keys, sorted_values = sorter.sort(keys, values, max_bits=max_bits)
	if values is None:
		return sorted_keys
	return sorted_keys, sorted_values