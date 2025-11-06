import unittest
from contextlib import contextmanager

import numpy as np
import TensorFrost as tf


def _should_skip_for_backend(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    keywords = (
        "not initialized",
        "glfw",
        "surface",
        "no suitable",
        "unavailable",
    )
    return any(token in message for token in keywords)


@contextmanager
def managed_window(width=320, height=240, title="ImGui Test Window"):
    try:
        win = tf.createWindow(width, height, title)
    except RuntimeError as exc:
        if _should_skip_for_backend(exc):
            raise unittest.SkipTest(f"Window backend unavailable: {exc}") from exc
        raise
    try:
        yield win
    finally:
        try:
            win.close()
        except Exception:
            pass


class ImGuiIntegrationTest(unittest.TestCase):
    def test_imgui_basic_widgets(self):
        with managed_window() as win:
            visible, open_flag = win.imgui_begin("Main Panel")
            self.assertTrue(visible, "ImGui window should be visible on begin")
            self.assertIsNone(open_flag, "Default begin call should return None for open flag")

            win.imgui_text("Hello from TensorFrost tests")

            button_pressed = win.imgui_button("Press Me")
            self.assertIn(button_pressed, (True, False))

            self.assertTrue(win.imgui_checkbox("Checkbox", True))

            slider_value = win.imgui_slider_float("Float Slider", 0.25, 0.0, 1.0)
            self.assertGreaterEqual(slider_value, 0.0)
            self.assertLessEqual(slider_value, 1.0)

            data = np.linspace(0.0, 1.0, 32, dtype=np.float32)
            win.imgui_plot_lines("Plot", data, overlay_text="test", graph_size=(120.0, 40.0))

            win.imgui_add_background_text("BG", (10.0, 10.0), (1.0, 1.0, 1.0, 1.0))
            win.imgui_scale_all_sizes(1.0)
            win.imgui_end()

            visible_secondary, open_flag_secondary = win.imgui_begin("Secondary", open=True)
            self.assertTrue(visible_secondary)
            self.assertIsInstance(open_flag_secondary, bool)
            win.imgui_text("Secondary window contents")
            updated_int = win.imgui_slider_int("Int Slider", 5, 0, 10)
            self.assertGreaterEqual(updated_int, 0)
            self.assertLessEqual(updated_int, 10)
            win.imgui_end()

            win.present()
            self.assertIsInstance(win.isOpen(), bool)


if __name__ == "__main__":
    unittest.main()
