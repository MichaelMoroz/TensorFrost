from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import TensorFrost as tf

try:
	from .sort import CountingSort
except ImportError:
	import sys

	_CURRENT_DIR = Path(__file__).resolve().parent
	if str(_CURRENT_DIR) not in sys.path:
		sys.path.insert(0, str(_CURRENT_DIR))

	from sort import CountingSort


_STAGE_NAMES = (
	"histogram",
	"block_sum",
	"block_prefix_stage1",
	"block_prefix_stage2",
	"scatter",
)


def _select_backend() -> None:
	if hasattr(tf, "initialize"):
		backend = getattr(tf, "vulkan", None)
		if backend is None:
			raise RuntimeError("TensorFrost Vulkan backend is unavailable on this build")
		tf.initialize(backend)


def _format_rate(value: float, unit: str) -> str:
	if value <= 0.0 or not math.isfinite(value):
		return f"0 {unit}"

	prefixes = ("", "K", "M", "G", "T", "P")
	magnitude = 0
	while value >= 1000.0 and magnitude < len(prefixes) - 1:
		value /= 1000.0
		magnitude += 1

	return f"{value:6.2f} {prefixes[magnitude]}{unit}"


def _format_stage_name(name: str) -> str:
	return name.replace("_", " ").title()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Interactive counting sort demo running continuously on the Vulkan backend.",
	)
	parser.add_argument("--size", type=int, default=1 << 22, help="Number of keys to sort")
	parser.add_argument(
		"--max-value",
		type=int,
		default=1 << 20,
		help="Exclusive upper bound for the generated key values",
	)
	parser.add_argument("--window-width", type=int, default=960, help="Window width in pixels")
	parser.add_argument("--window-height", type=int, default=540, help="Window height in pixels")
	parser.add_argument("--font-scale", type=float, default=1.6, help="Global ImGui font scale")
	parser.add_argument("--history", type=int, default=240, help="Number of samples stored for metrics history")
	parser.add_argument("--seed", type=int, default=1337, help="Seed for the random data generator")
	args = parser.parse_args()

	_select_backend()

	count = max(0, int(args.size))
	max_value = max(1, int(args.max_value))
	window_width = max(320, int(args.window_width))
	window_height = max(240, int(args.window_height))
	history_length = max(1, int(args.history))
	font_scale = max(0.1, float(args.font_scale)) if args.font_scale > 0 else 0.0

	rng = np.random.default_rng(int(args.seed))
	keys = (
		rng.integers(0, max_value, size=count, dtype=np.uint32)
		if count
		else np.empty(0, dtype=np.uint32)
	)
	values = (
		rng.integers(0, 1 << 31, size=count, dtype=np.uint32)
		if count
		else None
	)

	frame_times = deque(maxlen=history_length)
	sort_times = deque(maxlen=history_length)
	stage_totals_overall = {name: 0.0 for name in _STAGE_NAMES}

	validated = False
	validation_ok = False
	validation_message = ""

	sort_count = 0
	total_kernel_time = 0.0

	sorter = CountingSort(max_value=max_value)

	window_title = f"TensorFrost Counting Sort ({count:,} elements)"
	window = tf.createWindow(window_width, window_height, window_title)

	if font_scale > 0.0 and window is not None:
		window.imgui_scale_all_sizes(font_scale)
		window.imgui_set_font_global_scale(font_scale)

	start_time = time.perf_counter()
	last_frame_time = start_time

	while window is not None and window.isOpen():
		now = time.perf_counter()
		dt = now - last_frame_time
		last_frame_time = now

		if dt > 0.0:
			frame_times.append(dt)
		frame_time_total = sum(frame_times)
		fps = (len(frame_times) / frame_time_total) if frame_times and frame_time_total > 0.0 else 0.0

		do_validate = not validated
		return_arrays = do_validate
		_keys_out, _vals_out = sorter.sort(
			keys,
			values,
			collect_stage_timings=True,
			validate=do_validate,
			return_arrays=return_arrays,
		)

		stage_timings = sorter.last_stage_timings or {}
		total_pass_time = float(stage_timings.get("total_pass", 0.0))
		stage_sum_time = float(sum(v for k, v in stage_timings.items() if k != "total_pass"))
		kernel_time = total_pass_time if total_pass_time > 0.0 else stage_sum_time

		if sort_times.maxlen == len(sort_times):
			sort_times.popleft()
		sort_times.append(kernel_time)

		sort_count += 1
		total_kernel_time += kernel_time
		for name in _STAGE_NAMES:
			stage_totals_overall[name] += stage_timings.get(name, 0.0)

		if do_validate:
			errors = int(getattr(sorter, "last_validation_errors", 0) or 0)
			gpu_validation_ok = (errors == 0)

			cpu_validation_ok = True
			cpu_messages = []
			if return_arrays and count:
				reference_keys = np.sort(keys, kind="stable")
				if not np.array_equal(_keys_out, reference_keys):
					cpu_validation_ok = False
					cpu_messages.append("key mismatch")

			validation_ok = gpu_validation_ok and cpu_validation_ok
			if validation_ok:
				validation_message = "GPU + NumPy validation passed."
			else:
				failure_reasons = []
				if not gpu_validation_ok:
					failure_reasons.append(f"GPU reported {errors} out-of-order pairs")
				if not cpu_validation_ok:
					reason = ", ".join(cpu_messages) if cpu_messages else "NumPy comparison failed"
					failure_reasons.append(reason)
				validation_message = "Validation failed: " + "; ".join(failure_reasons)

			validated = True

		window_avg_sort = (sum(sort_times) / len(sort_times)) if sort_times else 0.0
		last_sort_ms = kernel_time * 1000.0
		avg_sort_ms = window_avg_sort * 1000.0

		last_sort_rate = (1.0 / kernel_time) if kernel_time > 0.0 else 0.0
		window_sort_rate = (1.0 / window_avg_sort) if window_avg_sort > 0.0 else 0.0
		overall_avg_sort = (total_kernel_time / sort_count) if sort_count else 0.0
		overall_sort_rate = (1.0 / overall_avg_sort) if overall_avg_sort > 0.0 else 0.0

		last_elements_per_sec = (count / kernel_time) if count and kernel_time > 0.0 else 0.0
		window_elements_per_sec = (count / window_avg_sort) if count and window_avg_sort > 0.0 else 0.0

		history_data = np.array(sort_times, dtype=np.float32) * 1000.0

		visible, _ = window.imgui_begin("Counting Sort Performance", open=None, flags=0)
		if visible:
			window.imgui_text(f"Elements: {count:,}")
			window.imgui_text(f"Max value: {max_value:,}")
			window.imgui_separator()
			window.imgui_text(f"Frame time: {dt * 1000.0:7.3f} ms")
			window.imgui_text(f"Average FPS: {fps:6.1f}")
			window.imgui_separator()

			if sort_count:
				window.imgui_text(f"Last kernel time: {last_sort_ms:7.3f} ms")
				if total_pass_time > 0.0:
					window.imgui_text(f"Total pass time (last): {total_pass_time * 1000.0:7.3f} ms")
				if window_avg_sort > 0.0:
					window.imgui_text(f"Window avg kernel: {avg_sort_ms:7.3f} ms")
				if overall_avg_sort > 0.0:
					window.imgui_text(f"Overall avg kernel: {overall_avg_sort * 1000.0:7.3f} ms")

				window.imgui_spacing()
				window.imgui_text(f"Sorts/sec (last): {last_sort_rate:8.2f}")
				window.imgui_text(f"Sorts/sec (window): {window_sort_rate:8.2f}")
				window.imgui_text(f"Sorts/sec (overall): {overall_sort_rate:8.2f}")

				if count:
					window.imgui_spacing()
					window.imgui_text(
						f"Elements/sec (last): {_format_rate(last_elements_per_sec, ' elements/s')}"
					)
					window.imgui_text(
						f"Elements/sec (window): {_format_rate(window_elements_per_sec, ' elements/s')}"
					)

				window.imgui_spacing()
				window.imgui_text(f"Total sorts: {sort_count:,}")
			else:
				if count:
					window.imgui_text("Waiting for first GPU sort...")
				else:
					window.imgui_text("No elements to sort.")

			if validated:
				color = (0.20, 0.80, 0.40, 1.0) if validation_ok else (0.92, 0.35, 0.32, 1.0)
				window.imgui_text_colored(color, validation_message)
			elif count:
				window.imgui_text("Validating against CPU reference...")

			if history_data.size:
				max_plot = float(np.max(history_data)) if history_data.size else 1.0
				max_plot = max(1.0, max_plot * 1.1)
				window.imgui_plot_lines(
					"Kernel time history (ms)",
					history_data,
					values_offset=0,
					overlay_text=f"{history_data.size} sample history",
					scale_min=0.0,
					scale_max=max_plot,
					graph_size=(0.0, 140.0),
					stride=4,
				)

			if stage_timings:
				window.imgui_separator()
				window.imgui_text("Kernel stages (last run):")
				for name, duration in sorted(stage_timings.items(), key=lambda item: item[1], reverse=True):
					if name == "total_pass":
						continue
					window.imgui_text(f"{_format_stage_name(name)}: {duration * 1000.0:7.3f} ms")

				window.imgui_spacing()
				window.imgui_text("Kernel stages (overall avg):")
				for name in sorted(stage_totals_overall.keys()):
					avg_duration = (stage_totals_overall[name] / sort_count) if sort_count else 0.0
					window.imgui_text(f"{_format_stage_name(name)}: {avg_duration * 1000.0:7.3f} ms")

		window.imgui_end()
		window.present()


if __name__ == "__main__":
	main()
