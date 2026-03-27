import unittest

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


class _ManagedWindow:
    def __init__(self, width=320, height=240, title="ImGui Test Window"):
        self._width = width
        self._height = height
        self._title = title
        self._window = None

    def __enter__(self):
        try:
            self._window = tf.createWindow(self._width, self._height, self._title)
        except RuntimeError as exc:
            if _should_skip_for_backend(exc):
                raise unittest.SkipTest(f"Window backend unavailable: {exc}") from exc
            raise
        return self._window

    def __exit__(self, exc_type, exc, tb):
        if self._window is not None:
            try:
                self._window.close()
            except Exception:
                pass
            self._window = None
        return False


def managed_window(width=320, height=240, title="ImGui Test Window"):
    return _ManagedWindow(width, height, title)


class ImGuiIntegrationTest(unittest.TestCase):
    def test_imgui_basic_widgets(self):
        with managed_window() as win:
            if win.imgui_begin_main_menu_bar():
                if win.imgui_begin_menu("Root"):
                    self.assertIn(win.imgui_menu_item("Item"), (True, False))
                    win.imgui_end_menu()
                win.imgui_end_main_menu_bar()

            visible, open_flag = win.imgui_begin("Main Panel")
            self.assertTrue(visible, "ImGui window should be visible on begin")
            self.assertIsNone(open_flag, "Default begin call should return None for open flag")

            win.imgui_text("Hello from TensorFrost tests")
            win.imgui_same_line()
            win.imgui_text("Inline")
            win.imgui_spacing()
            win.imgui_separator()
            win.imgui_indent()
            child_visible = win.imgui_begin_child("child", size=(120.0, 60.0), border=False)
            self.assertIsInstance(child_visible, bool)
            if child_visible:
                win.imgui_text_wrapped("Wrapped text inside child region to check bindings work as expected.")
                win.imgui_bullet_text("Bullet item content")
            win.imgui_end_child()
            win.imgui_unindent()
            win.imgui_text_colored((1.0, 0.0, 0.0, 1.0), "Colored text")
            alpha_scale = win.imgui_get_font_global_scale()
            win.imgui_set_font_global_scale(alpha_scale)
            style_color = win.imgui_get_style_color_vec4(0)
            self.assertEqual(len(style_color), 4)
            win.imgui_push_style_color(0, style_color)
            win.imgui_push_style_var_float(0, 0.95)
            win.imgui_push_style_var_vec2(2, (4.0, 4.0))
            win.imgui_pop_style_var(2)
            win.imgui_pop_style_color()
            win.imgui_set_style_color_vec4(0, style_color)

            button_pressed = win.imgui_button("Press Me")
            self.assertIn(button_pressed, (True, False))

            self.assertTrue(win.imgui_checkbox("Checkbox", True))

            slider_value = win.imgui_slider_float("Float Slider", 0.25, 0.0, 1.0)
            self.assertGreaterEqual(slider_value, 0.0)
            self.assertLessEqual(slider_value, 1.0)

            data = np.linspace(0.0, 1.0, 32, dtype=np.float32)
            win.imgui_plot_lines("Plot", data, overlay_text="test", graph_size=(120.0, 40.0))

            text_changed, text_value = win.imgui_input_text("Input", "hello")
            self.assertIsInstance(text_changed, bool)
            self.assertIsInstance(text_value, str)

            updated_int_input = win.imgui_input_int("Input Int", 10)
            self.assertIsInstance(updated_int_input, int)

            updated_float_input = win.imgui_input_float("Input Float", 3.14)
            self.assertIsInstance(updated_float_input, float)

            color_changed3, rgb = win.imgui_color_edit3("Color3", (0.1, 0.2, 0.3))
            self.assertIsInstance(color_changed3, bool)
            self.assertEqual(len(rgb), 3)

            color_changed4, rgba = win.imgui_color_edit4("Color4", (0.1, 0.2, 0.3, 0.4))
            self.assertIsInstance(color_changed4, bool)
            self.assertEqual(len(rgba), 4)

            if win.imgui_begin_menu_bar():
                if win.imgui_begin_menu("File"):
                    self.assertIn(win.imgui_menu_item("New"), (True, False))
                    win.imgui_end_menu()
                win.imgui_end_menu_bar()

            win.imgui_open_popup("ContextPopup")
            if win.imgui_begin_popup("ContextPopup"):
                win.imgui_text("Popup body")
                win.imgui_close_current_popup()
                win.imgui_end_popup()

            win.imgui_open_popup("ModalPopup")
            visible_modal, modal_open = win.imgui_begin_popup_modal("ModalPopup", open=True)
            self.assertIn(visible_modal, (True, False))
            self.assertIsInstance(modal_open, bool)
            if visible_modal:
                win.imgui_text("Modal body")
                win.imgui_close_current_popup()
                win.imgui_end_popup()

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
