import unittest

import numpy as np

import TensorFrost as tf


_SIMPLE_SLANG = r"""
[[vk::binding(0,0)]] RWStructuredBuffer<uint> Pixels : register(u0, space0);

[numthreads(64,1,1)]
void csMain(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= Pixels.length()) return;
    Pixels[idx] = 0xff3366ff;
}
"""


class VulkanWindowTest(unittest.TestCase):
    def test_compute_dispatch_and_window_present(self):
        width = height = 8
        thread_count = width * height
        local_size = 64
        group_count = max((thread_count + local_size - 1) // local_size, 1)

        try:
            pixel_buffer = tf.createBuffer(thread_count, 4, False)
        except RuntimeError as exc:  # pragma: no cover - Vulkan not available
            self.skipTest(f"Vulkan buffer creation failed: {exc}")

        program = None
        try:
            program = tf.createComputeProgramFromSlang(
                "window_test_fill", _SIMPLE_SLANG, "csMain", ro_count=0, rw_count=1
            )
        except RuntimeError as exc:  # pragma: no cover - Vulkan program creation failed
            self.skipTest(f"Vulkan program creation failed: {exc}")

        window = None
        program.run([], [pixel_buffer], group_count)
        pixels = pixel_buffer.getData(np.dtype(np.uint32), thread_count)
        self.assertTrue(np.all(pixels == 0xFF3366FF), "Compute shader did not write expected color")

        try:
            window = tf.createWindow(width, height, "TensorFrost Vulkan Test")
        except RuntimeError as exc:  # pragma: no cover - Vulkan window not available
            self.skipTest(f"Vulkan window creation failed: {exc}")

        # Present once to ensure the binding path is exercised.
        window.drawBuffer(pixel_buffer, width, height)
        self.assertTrue(window.isOpen(), "Window should report as open after initial present")

        if window is not None:
            window.close()
        window = None
        program = None
        pixel_buffer = None

        # Ensure GPU resources are released before interpreter shutdown.
        import gc

        gc.collect()


if __name__ == "__main__":
    unittest.main()
