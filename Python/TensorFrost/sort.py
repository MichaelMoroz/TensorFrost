from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from importlib import resources
from typing import Dict, Optional, Tuple

import numpy as np

from . import TensorFrost as tf

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


def _load_shader_source(filename: str) -> str:
	package = f"{__package__}.shaders.radix"
	try:
		return resources.files(package).joinpath(filename).read_text(encoding="utf-8")  # type: ignore[attr-defined]
	except AttributeError:
		return resources.read_text(package, filename)


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
		scatter_source = f"#define TF_HISTOGRAM_SIZE {self.histogram_size}u\n" + _load_shader_source("scatter.slang")
		self._scatter_program = tf.createComputeProgramFromSlang(
			"radix_scatter",
			scatter_source,
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

			map_groups = _dispatch_groups(element_count, self.group_size)
			reduction_group_size = 64
			unpack_groups = _dispatch_groups(histogram_size * num_groups, reduction_group_size)
			prefix_local_groups = _dispatch_groups(histogram_size * block_count, reduction_group_size)
			prefix_block_groups = _dispatch_groups(histogram_size, reduction_group_size)
			prefix_accum_groups = _dispatch_groups(histogram_size * block_count, reduction_group_size)
			bucket_scan_groups = _dispatch_groups(histogram_size, reduction_group_size)
			scatter_groups = num_groups
			histogram_groups = num_groups

			self._map_to_uint_program.run(
				[map_buffer, key_buffers[0]],
				[key_buffers[1]],
				map_groups,
			)

			key_in = key_buffers[1]
			key_out = key_buffers[0]
			val_in, val_out = value_buffers

			for pass_index in range(passes):
				params_array[2] = np.uint32(pass_index * self.bits_per_pass)
				params_buffer.setData(params_array)

				self._histogram_program.run(
					[params_buffer, key_in],
					[packed_hist_buffer],
					histogram_groups,
				)

				self._unpack_program.run(
					[params_buffer, packed_hist_buffer],
					[group_hist_buffer],
					unpack_groups,
				)

				self._prefix_local_program.run(
					[params_buffer, group_hist_buffer],
					[prefix_buffer, block_totals_buffer],
					prefix_local_groups,
				)

				self._prefix_blocks_program.run(
					[params_buffer, block_totals_buffer],
					[block_prefix_buffer],
					prefix_block_groups,
				)

				self._prefix_accum_program.run(
					[params_buffer, block_prefix_buffer],
					[prefix_buffer],
					prefix_accum_groups,
				)

				self._bucket_scan_program.run(
					[params_buffer, prefix_buffer],
					[bucket_scan_buffer],
					bucket_scan_groups,
				)

				self._scatter_program.run(
					[params_buffer, key_in, val_in, prefix_buffer, bucket_scan_buffer],
					[key_out, val_out],
					scatter_groups,
				)

				key_in, key_out = key_out, key_in
				if values_array is not None:
					val_in, val_out = val_out, val_in

			self._map_from_uint_program.run(
				[map_buffer, key_in],
				[key_out],
				map_groups,
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
# def radix(keys, values = None, bits_per_pass = 6, max_bits = 32):
#     def prefix_sum_grouped(A, axis = -1):
#         axis = len(A.shape) + axis if axis < 0 else axis
#         group_size = 64
#         grouped = tf.split_dim(A, group_size, axis)
#         group_scan = tf.prefix_sum(tf.sum(grouped, axis = axis + 1), axis = axis)
#         ids = grouped.indices
#         gid, eid = ids[axis], ids[axis + 1]
#         ids = [ids[i] for i in range(len(ids)) if i != axis + 1]
#         ids[axis] = gid - 1
#         group_scan = tf.prefix_sum(grouped + tf.select((gid == 0) | (eid != 0), tf.uint(0), group_scan[tuple(ids)]), axis = axis + 1)
#         full_scan = tf.merge_dim(group_scan, target_size = A.shape[axis], axis = axis + 1)
#         return full_scan
#
#     sign_bit = ~tf.uint(0x7FFFFFFF)
#
#     def map_float_to_uint(x):
#         # Convert float to uint representation
#         ux = tf.asuint(x)
#         # Compute mask
#         mask = tf.select((ux >> 31) == 1, ~tf.uint(0), sign_bit)
#         # Apply XOR
#         return ux ^ mask
#
#     def map_uint_to_float(x):
#         # Compute mask
#         mask = tf.select((x >> 31) == 0, ~tf.uint(0), sign_bit)
#         # Apply XOR and convert back to float
#         return tf.asfloat(x ^ mask)
#
#     def map_int_to_uint(x):
#         return tf.asuint(x) ^ sign_bit
#
#     def map_uint_to_int(x):
#         return tf.asint(x ^ sign_bit)
#
#     tf.region_begin('Radix sort')
#
#     has_values = values is not None
#
#     keys = tf.copy(keys)
#     if has_values:
#         values = tf.copy(values)
#
#     original_type = keys.type
#     if(original_type == tf.float32):
#         keys = map_float_to_uint(keys)
#
#     if(original_type == tf.int32):
#         keys = map_int_to_uint(keys)
#
#     iters = (max_bits + bits_per_pass - 1) // bits_per_pass
#     group_size = 128
#     histogram_size = 2 ** bits_per_pass
#
#     def GetBits(A, i):
#         return (A >> (i * bits_per_pass)) & tf.uint(histogram_size - 1)
#
#     keys1 = tf.buffer(keys.shape, keys.type)
#     values1 = None
#
#     if has_values:
#         values1 = tf.buffer(values.shape, values.type)
#
#     with tf.loop(iters // 2) as iter:
#         def SortIteration(keys_in, keys_out, values_in, values_out, iter):
#             tf.region_begin('Radix sort iteration')
#             grouped = tf.split_dim(GetBits(keys_in, iter), group_size)
#
#             # Do a packed histogram, since we sum 128 elements at a time, we can pack 4 values into a single uint32
#             g, e, i = tf.indices([grouped.shape[0], grouped.shape[1], tf.int(histogram_size/4)])
#             this_key = grouped[g, e]
#             packed_is_bit = (tf.uint(this_key == tf.uint(4*i))) + (tf.uint(this_key == tf.uint(4*i+1)) << 8) + (tf.uint(this_key == tf.uint(4*i+2)) << 16) + (tf.uint(this_key == tf.uint(4*i+3)) << 24)
#             packed_is_bit = tf.select((g*group_size + e) < keys_in.shape[0], packed_is_bit, tf.uint(0))
#             group_histogram_packed = tf.sum(packed_is_bit, axis = 1)
#
#             g, i = tf.indices([grouped.shape[0], histogram_size])
#             group_histogram = tf.uint((group_histogram_packed[g, i / 4] >> (8*(i % 4))) & tf.uint(0xFF))
#
#             group_histogram_scan = prefix_sum_grouped(group_histogram, axis = 0)
#             i, = tf.indices([histogram_size])
#             total_bit_histogram = tf.prefix_sum(group_histogram_scan[group_histogram_scan.shape[0] - 1, i])
#
#             with tf.kernel(grouped.shape, group_size=[group_size]) as (g, e):
#                 if(tf.current_backend() == tf.cpu): #dont use group barriers on CPU - doesn't work
#                     element = g * group_size + e
#                     with tf.if_cond(element < keys_in.shape[0]):
#                         old_key = keys_in[element]
#                         old_val = values_in[element]
#                         bit = GetBits(old_key, iter)
#                         total_offset = tf.select(g == 0, tf.uint(0), group_histogram_scan[g - 1, bit]) + tf.select(bit == tf.uint(0), tf.uint(0), total_bit_histogram[bit - tf.uint(1)])
#                         with tf.loop(e) as j:
#                             total_offset.val += tf.uint(grouped[g, j] == bit)
#                         keys_out[total_offset] = old_key
#                         values_out[total_offset] = old_val
#                 else:
#                     temp = tf.group_buffer(group_size, tf.uint32)
#                     half_count = tf.group_buffer(histogram_size, tf.uint32)
#                     gtid = g.block_thread_index(0)
#
#                     #initialize counters
#                     for i in range((histogram_size + group_size - 1) // group_size):
#                         index = gtid + i * group_size
#                         with tf.if_cond(index < histogram_size):
#                             half_count[index] = 0
#                     tf.group_barrier()
#
#                     element = g * group_size + e
#                     with tf.if_cond(element < keys_in.shape[0]):
#                         old_key = keys_in[element]
#                         bit = GetBits(old_key, iter)
#                         temp[gtid] = bit
#
#                         #count number of bits set in previous sub groups
#                         quarter_index = e / (group_size // 4)
#                         with tf.if_cond(quarter_index < 3):
#                             tf.scatterAdd(half_count[bit], tf.uint(quarter_index < 1) | (tf.uint(quarter_index < 2) << 8) | (tf.uint(quarter_index < 3) << 16))
#
#                         tf.group_barrier()
#
#                         if has_values:
#                             old_val = values_in[element]
#
#                         total_offset = tf.select(g == 0, tf.uint(0), group_histogram_scan[g - 1, tf.int(bit)]) + tf.select(tf.int(bit) == 0, tf.uint(0), total_bit_histogram[tf.int(bit) - 1])
#                         total_offset += tf.select(quarter_index > 0, (half_count[bit] >> (8*(quarter_index-1))) & tf.uint(0xFF), tf.uint(0))
#                         begin_index = quarter_index * (group_size // 4)
#                         with tf.loop(begin_index, e) as j:
#                             total_offset.val += tf.uint(temp[j] == bit)
#                         keys_out[total_offset] = old_key
#
#                         if has_values:
#                             values_out[total_offset] = old_val
#
#             tf.region_end('Radix sort iteration')
#
#         SortIteration(keys, keys1, values, values1, 2 * iter)
#         SortIteration(keys1, keys, values1, values, 2 * iter + 1)
#
#     tf.region_end('Radix sort')
#
#     if(original_type == tf.float32):
#         keys = map_uint_to_float(keys)
#
#     if(original_type == tf.int32):
#         keys = map_uint_to_int(keys)
#
#     if has_values:
#         return keys, values
#     else:
#         return keys
