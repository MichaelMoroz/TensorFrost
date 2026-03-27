from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import TensorFrost as tf

__all__ = ["CountingSort", "counting_sort"]


def _dispatch_groups(work_items: int, threads_per_group: int) -> int:
	if work_items <= 0:
		return 0
	return (work_items + threads_per_group - 1) // threads_per_group


def _prepare_keys(keys: np.ndarray) -> np.ndarray:
	array = np.asarray(keys)
	if array.ndim != 1:
		raise ValueError("counting_sort expects a 1D array of keys")
	if array.dtype != np.uint32:
		array = array.astype(np.uint32, copy=False)
	return array


def _prepare_values(values: np.ndarray) -> Tuple[np.ndarray, np.dtype]:
	array = np.asarray(values)
	if array.ndim != 1:
		raise ValueError("counting_sort expects a 1D array of values when provided")
	dtype = array.dtype
	if dtype not in (np.uint32, np.int32, np.float32):
		raise TypeError("values must have dtype uint32, int32, or float32")
	return array, dtype


_SHADER_DIR = Path(__file__).resolve().parent / "shaders"


def _load_shader_source(filename: str) -> str:
	return (_SHADER_DIR / filename).read_text(encoding="utf-8")


@dataclass(frozen=True)
class _SorterKey:
	max_value: int
	block_span: int
	group_size: int
	blocks_per_group: int


class CountingSort:
	"""GPU counting sort built on top of TensorFrost compute and Slang shaders."""

	def __init__(
		self,
		*,
		max_value: int,
		block_span: int = 256,
		group_size: int = 128,
		blocks_per_group: int = 256,
	) -> None:
		if max_value <= 0:
			raise ValueError("max_value must be positive")
		if block_span <= 0:
			raise ValueError("block_span must be positive")
		if group_size <= 0 or group_size > 1024:
			raise ValueError("group_size must be within (0, 1024]")
		if blocks_per_group <= 0:
			raise ValueError("blocks_per_group must be positive")

		self.value_count = int(max_value)
		self.block_span = int(block_span)
		self.group_size = int(group_size)
		self.blocks_per_group = int(blocks_per_group)
		self.block_count = max((self.value_count + self.block_span - 1) // self.block_span, 1)
		self.block_group_count = max((self.block_count + self.blocks_per_group - 1) // self.blocks_per_group, 1)

		self.last_stage_timings: Optional[Dict[str, float]] = None
		self.last_validation_errors: Optional[int] = None

		def inject_defines(filename: str) -> str:
			defines = [
				f"#define CS_GROUP_SIZE {self.group_size}u",
				f"#define CS_BLOCK_SPAN {self.block_span}u",
			]
			return "\n".join(defines) + "\n" + _load_shader_source(filename)

		self._dummy_values_buffer = tf.createBuffer(1, 4, False)

		self._histogram_program = tf.createComputeProgramFromSlang(
			"count_histogram_rank",
			inject_defines("histogram_rank.slang"),
			"csHistogramAndRank",
			ro_count=1,
			rw_count=2,
			push_constant_size=8,
		)
		self._segment_scan_program = tf.createComputeProgramFromSlang(
			"count_segment_scan",
			inject_defines("block_sum.slang"),
			"csSegmentScan",
			ro_count=1,
			rw_count=2,
			push_constant_size=12,
		)
		self._block_prefix_stage2_program = tf.createComputeProgramFromSlang(
			"count_block_prefix_stage2",
			inject_defines("block_prefix_stage2.slang"),
			"csBlockPrefixStage2",
			ro_count=1,
			rw_count=1,
			push_constant_size=4,
		)
		self._scatter_program = tf.createComputeProgramFromSlang(
			"count_scatter",
			inject_defines("scatter.slang"),
			"csScatter",
			ro_count=6,
			rw_count=2,
			push_constant_size=20,
		)
		self._validate_program = tf.createComputeProgramFromSlang(
			"count_validate",
			inject_defines("validate_sorted.slang"),
			"csValidate",
			ro_count=1,
			rw_count=1,
			push_constant_size=4,
		)

	def sort(
		self,
		keys: np.ndarray,
		values: Optional[np.ndarray] = None,
		*,
		collect_stage_timings: bool = False,
		validate: bool = False,
		return_arrays: bool = False,
	) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		keys_array = _prepare_keys(keys)
		element_count = int(keys_array.shape[0])

		if values is not None:
			values_array, values_dtype = _prepare_values(values)
			if values_array.shape[0] != element_count:
				raise ValueError("values must have the same length as keys")
		else:
			values_array = None
			values_dtype = None

		if element_count == 0:
			self.last_stage_timings = {} if collect_stage_timings else None
			if validate:
				self.last_validation_errors = 0
			if values_array is None:
				return keys_array.copy(), None
			return keys_array.copy(), values_array.copy()

		if np.max(keys_array) >= self.value_count:
			raise ValueError(f"keys must be within [0, {self.value_count}) for this sorter")

		if collect_stage_timings:
			stage_totals: Dict[str, float] = {
				"histogram": 0.0,
				"block_sum": 0.0,
				"block_prefix_stage1": 0.0,
				"block_prefix_stage2": 0.0,
				"scatter": 0.0,
			}
		else:
			stage_totals = {}

		key_in_buffer = tf.createBuffer(max(element_count, 1), 4, False)
		key_in_buffer.setData(keys_array)
		key_out_buffer = tf.createBuffer(max(element_count, 1), 4, False)

		if values_array is not None:
			val_in_buffer = tf.createBuffer(max(element_count, 1), 4, False)
			val_in_buffer.setData(values_array)
			val_out_buffer = tf.createBuffer(max(element_count, 1), 4, False)
		else:
			dummy = self._dummy_values_buffer
			val_in_buffer = dummy
			val_out_buffer = dummy

		histogram_buffer = tf.createBuffer(max(self.value_count, 1), 4, False)
		z_hist = np.zeros(self.value_count, dtype=np.uint32)
		histogram_buffer.setData(z_hist)

		local_rank_buffer = tf.createBuffer(max(element_count, 1), 4, False)
		local_prefix_buffer = tf.createBuffer(max(self.value_count, 1), 4, False)
		block_totals_buffer = tf.createBuffer(max(self.block_count, 1), 4, False)
		block_prefix_stage1_buffer = tf.createBuffer(max(self.block_count, 1), 4, False)
		block_group_totals_buffer = tf.createBuffer(max(self.block_group_count, 1), 4, False)
		block_group_prefix_buffer = tf.createBuffer(max(self.block_group_count, 1), 4, False)

		hist_params = np.zeros(2, dtype=np.uint32)
		hist_params[0] = np.uint32(element_count)
		hist_params[1] = np.uint32(self.value_count)

		segment_params_hist = np.zeros(3, dtype=np.uint32)
		segment_params_hist[0] = np.uint32(self.block_count)
		segment_params_hist[1] = np.uint32(self.block_span)
		segment_params_hist[2] = np.uint32(self.value_count)

		stage1_params = np.zeros(3, dtype=np.uint32)
		stage1_params[0] = np.uint32(self.block_group_count)
		stage1_params[1] = np.uint32(self.blocks_per_group)
		stage1_params[2] = np.uint32(self.block_count)

		stage2_params = np.zeros(1, dtype=np.uint32)
		stage2_params[0] = np.uint32(self.block_group_count)

		scatter_params = np.zeros(5, dtype=np.uint32)
		scatter_params[0] = np.uint32(element_count)
		scatter_params[1] = np.uint32(self.value_count)
		scatter_params[2] = np.uint32(self.block_span)
		scatter_params[3] = np.uint32(self.blocks_per_group)
		scatter_params[4] = np.uint32(1 if values_array is not None else 0)

		histogram_groups = _dispatch_groups(element_count, self.group_size)
		block_sum_groups = _dispatch_groups(self.block_count, 64)
		block_prefix_stage1_groups = _dispatch_groups(self.block_group_count, 64)
		block_prefix_stage2_groups = _dispatch_groups(self.block_group_count, 64)
		scatter_groups = _dispatch_groups(element_count, self.group_size)

		total_start = time.perf_counter() if collect_stage_timings else None

		start = time.perf_counter() if collect_stage_timings else None
		self._histogram_program.run(
			[key_in_buffer],
			[histogram_buffer, local_rank_buffer],
			histogram_groups,
			hist_params,
		)
		if collect_stage_timings and start is not None:
			stage_totals["histogram"] += time.perf_counter() - start

		start = time.perf_counter() if collect_stage_timings else None
		self._segment_scan_program.run(
			[histogram_buffer],
			[local_prefix_buffer, block_totals_buffer],
			block_sum_groups,
			segment_params_hist,
		)
		if collect_stage_timings and start is not None:
			stage_totals["block_sum"] += time.perf_counter() - start

		start = time.perf_counter() if collect_stage_timings else None
		self._segment_scan_program.run(
			[block_totals_buffer],
			[block_prefix_stage1_buffer, block_group_totals_buffer],
			block_prefix_stage1_groups,
			stage1_params,
		)
		if collect_stage_timings and start is not None:
			stage_totals["block_prefix_stage1"] += time.perf_counter() - start

		start = time.perf_counter() if collect_stage_timings else None
		self._block_prefix_stage2_program.run(
			[block_group_totals_buffer],
			[block_group_prefix_buffer],
			block_prefix_stage2_groups,
			stage2_params,
		)
		if collect_stage_timings and start is not None:
			stage_totals["block_prefix_stage2"] += time.perf_counter() - start

		start = time.perf_counter() if collect_stage_timings else None
		self._scatter_program.run(
			[
				key_in_buffer,
				val_in_buffer,
				local_rank_buffer,
				local_prefix_buffer,
				block_prefix_stage1_buffer,
				block_group_prefix_buffer,
			],
			[key_out_buffer, val_out_buffer],
			scatter_groups,
			scatter_params,
		)
		if collect_stage_timings and start is not None:
			stage_totals["scatter"] += time.perf_counter() - start

		if collect_stage_timings and total_start is not None:
			stage_totals["total_pass"] = time.perf_counter() - total_start

		if validate:
			validate_params = np.zeros(1, dtype=np.uint32)
			validate_params[0] = np.uint32(element_count)

			error_buf = tf.createBuffer(1, 4, False)
			error_zero = np.zeros(1, dtype=np.uint32)
			error_buf.setData(error_zero)

			self._validate_program.run(
				[key_out_buffer],
				[error_buf],
				_dispatch_groups(element_count, self.group_size),
				validate_params,
			)

			error_count = int(error_buf.getData(np.dtype(np.uint32), 1)[0])
			self.last_validation_errors = error_count
		else:
			self.last_validation_errors = None

		if return_arrays:
			sorted_keys = key_out_buffer.getData(np.dtype(np.uint32), element_count)
			if values_array is not None and values_dtype is not None:
				sorted_values = val_out_buffer.getData(values_dtype, element_count)
			else:
				sorted_values = None
		else:
			sorted_keys = np.empty(0, dtype=np.uint32)
			sorted_values = None if values_array is None or values_dtype is None else np.empty(0, dtype=values_dtype)

		self.last_stage_timings = stage_totals if collect_stage_timings else None
		return sorted_keys, sorted_values


_SORTER_CACHE: Dict[_SorterKey, CountingSort] = {}


def _get_sorter(max_value: int, block_span: int, group_size: int, blocks_per_group: int) -> CountingSort:
	key = _SorterKey(max_value, block_span, group_size, blocks_per_group)
	sorter = _SORTER_CACHE.get(key)
	if sorter is None:
		sorter = CountingSort(
			max_value=max_value,
			block_span=block_span,
			group_size=group_size,
			blocks_per_group=blocks_per_group,
		)
		_SORTER_CACHE[key] = sorter
	return sorter


def counting_sort(
	keys: np.ndarray,
	values: Optional[np.ndarray] = None,
	*,
	max_value: int,
	block_span: int = 256,
	group_size: int = 128,
	blocks_per_group: int = 256,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
	"""Convenience wrapper around :class:`CountingSort`."""

	if max_value is None:
		raise ValueError("max_value must be provided for counting_sort")

	sorter = _get_sorter(int(max_value), int(block_span), int(group_size), int(blocks_per_group))
	sorted_keys, sorted_values = sorter.sort(keys, values)
	if values is None:
		return sorted_keys
	return sorted_keys, sorted_values
