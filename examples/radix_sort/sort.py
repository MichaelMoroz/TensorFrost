from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import TensorFrost as tf

__all__ = ["HistogramRadixSort", "radix_sort"]

_TYPE_CODES: Dict[str, np.uint32] = {
	"uint": np.uint32(0),
	"int": np.uint32(1),
	"float": np.uint32(2),
}


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

		self._map_to_uint_program = tf.createComputeProgramFromSlang(
			"radix_map_to_uint",
			_load_shader_source("map_to_uint.slang"),
			"csMapToUint",
			ro_count=2,
			rw_count=1,
		)
		self._map_from_uint_program = tf.createComputeProgramFromSlang(
			"radix_map_from_uint",
			_load_shader_source("map_from_uint.slang"),
			"csMapFromUint",
			ro_count=2,
			rw_count=1,
		)

		self._histogram_program = tf.createComputeProgramFromSlang(
			"radix_histogram",
			_load_shader_source("histogram.slang"),
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
		self._scatter_program = tf.createComputeProgramFromSlang(
			"radix_scatter",
			_load_shader_source("scatter.slang"),
			"csScatter",
			ro_count=5,
			rw_count=2,
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
			if values_array is None:
				return empty_keys, None
			return empty_keys, values_array.copy()

		max_bits = int(min(max_bits, 32))
		histogram_size = 1 << self.bits_per_pass
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

		with ExitStack() as stack:
			params_buffer = tf.createBuffer(params_array.size, 4, True)
			stack.callback(params_buffer.release)
			params_buffer.setData(params_array)

			map_buffer = tf.createBuffer(map_params.size, 4, True)
			stack.callback(map_buffer.release)
			map_buffer.setData(map_params)

			key_buffers = [tf.createBuffer(max(element_count, 1), 4, False) for _ in range(2)]
			for buf in key_buffers:
				stack.callback(buf.release)
			key_buffers[0].setData(keys_array)

			if values_array is not None:
				value_buffers = [tf.createBuffer(max(element_count, 1), 4, False) for _ in range(2)]
				for buf in value_buffers:
					stack.callback(buf.release)
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

			for buf in (
				packed_hist_buffer,
				group_hist_buffer,
				prefix_buffer,
				block_totals_buffer,
				block_prefix_buffer,
				bucket_scan_buffer,
			):
				stack.callback(buf.release)

			self._map_to_uint_program.run(
				[map_buffer, key_buffers[0]],
				[key_buffers[1]],
				element_count,
			)

			key_in = key_buffers[1]
			key_out = key_buffers[0]
			val_in, val_out = value_buffers

			for pass_index in range(passes):
				params_array[2] = np.uint32(pass_index * self.bits_per_pass)
				params_buffer.setData(params_array)

				dispatch_threads = num_groups * self.group_size
				self._histogram_program.run(
					[params_buffer, key_in],
					[packed_hist_buffer],
					dispatch_threads,
				)

				self._unpack_program.run(
					[params_buffer, packed_hist_buffer],
					[group_hist_buffer],
					histogram_size * num_groups,
				)

				self._prefix_local_program.run(
					[params_buffer, group_hist_buffer],
					[prefix_buffer, block_totals_buffer],
					histogram_size * block_count,
				)

				self._prefix_blocks_program.run(
					[params_buffer, block_totals_buffer],
					[block_prefix_buffer],
					histogram_size,
				)

				self._prefix_accum_program.run(
					[params_buffer, block_prefix_buffer],
					[prefix_buffer],
					histogram_size * block_count,
				)

				self._bucket_scan_program.run(
					[params_buffer, prefix_buffer],
					[bucket_scan_buffer],
					histogram_size,
				)

				self._scatter_program.run(
					[params_buffer, key_in, val_in, prefix_buffer, bucket_scan_buffer],
					[key_out, val_out],
					dispatch_threads,
				)

				key_in, key_out = key_out, key_in
				if values_array is not None:
					val_in, val_out = val_out, val_in

			self._map_from_uint_program.run(
				[map_buffer, key_in],
				[key_out],
				element_count,
			)

			sorted_keys = key_out.getData(key_dtype, element_count)
			if values_array is not None and values_dtype is not None:
				sorted_values = val_in.getData(values_dtype, element_count)
			else:
				sorted_values = None

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