from __future__ import annotations

import argparse
import time
from contextlib import ExitStack

import numpy as np
import TensorFrost as tf

try:
    from .sort import HistogramRadixSort
except ImportError:
    import sys
    from pathlib import Path

    _CURRENT_DIR = Path(__file__).resolve().parent
    if str(_CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(_CURRENT_DIR))

    from sort import HistogramRadixSort


def _select_backend() -> None:
    if hasattr(tf, "initialize"):
        backend = getattr(tf, "vulkan", None)
        if backend is None:
            raise RuntimeError("TensorFrost Vulkan backend is unavailable on this build")
        tf.initialize(backend)


def main() -> None:
    parser = argparse.ArgumentParser(description="Histogram radix sort demo running on the Vulkan backend.")
    parser.add_argument("--size", type=int, default=1 << 20, help="Number of key/value pairs to sort")
    parser.add_argument("--bits", type=int, default=6, help="Bits processed per pass")
    args = parser.parse_args()

    _select_backend()

    count = max(0, int(args.size))
    bits_per_pass = max(1, int(args.bits))
    rng = np.random.default_rng(1234)

    keys = rng.standard_normal(count, dtype=np.float32)
    values = rng.integers(0, 1 << 31, size=count, dtype=np.uint32)

    with ExitStack() as stack:
        renderdoc_is_available = getattr(tf, "renderdoc_is_available", None)
        renderdoc_start = getattr(tf, "renderdoc_start_capture", None)
        renderdoc_end = getattr(tf, "renderdoc_end_capture", None)
        if (
            callable(renderdoc_is_available)
            and renderdoc_is_available()
            and callable(renderdoc_start)
            and callable(renderdoc_end)
        ):
            renderdoc_start()
            print("RenderDoc capture started")
            stack.callback(renderdoc_end)

        sorter = HistogramRadixSort(bits_per_pass=bits_per_pass)
        stack.callback(sorter.close)
        start_time = time.perf_counter()
        sorted_keys, sorted_values = sorter.sort(keys, values)
        elapsed = time.perf_counter() - start_time

    if sorted_values is None:
        sorted_values = np.empty_like(values)

    order = np.argsort(keys, kind="stable")
    reference_keys = keys[order]
    reference_values = values[order]

    key_match = np.allclose(sorted_keys, reference_keys, atol=0.0, rtol=0.0)
    value_match = np.array_equal(sorted_values, reference_values)

    print(f"Sorted {count} elements with bits_per_pass={bits_per_pass}")
    print(f"Sort elapsed: {elapsed * 1e3:.3f} ms ({elapsed:.6f} s)")
    print(f"Keys match reference: {key_match}")
    print(f"Values match reference: {value_match}")
    if count:
        preview = min(10, count)
        print("First few sorted keys:")
        print(sorted_keys[:preview])


if __name__ == "__main__":
    main()
