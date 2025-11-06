"""Interactive ImGui showcase demonstrating the TensorFrost Vulkan window helpers.

This example exercises a large portion of the Python ImGui bindings that are
exposed through ``TensorFrost.Window``.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
import TensorFrost as tf

# ImGui enums we need inside the sample. They mirror the values from imgui.h.
IMGUI_WINDOW_FLAGS_MENU_BAR = 1 << 3
IMGUI_COL_WINDOW_BG = 2
IMGUI_COL_BUTTON = 21
IMGUI_COL_BUTTON_HOVERED = 22
IMGUI_STYLEVAR_WINDOW_PADDING = 2
IMGUI_STYLEVAR_FRAME_ROUNDING = 12
IMGUI_STYLEVAR_ITEM_SPACING = 14


def main() -> None:
    width, height = 960, 600
    window = tf.createWindow(width, height, "TensorFrost ImGui Showcase")

    sample_history: deque[float] = deque(maxlen=512)
    frame_times: deque[float] = deque(maxlen=240)

    start_time = time.perf_counter()
    last_time = start_time
    total_time = 0.0

    state = {
        "animate": True,
        "show_plot": True,
        "wave_speed": 1.0,
        "wave_scale": 1.0,
        "sample_count": 180,
        "greeting": "Hello from TensorFrost!",
        "accent": (0.25, 0.62, 0.98, 1.0),
    "theme": "dark",
    "font_scale": 2.0,
    }

    themes = {
        "dark": {
            IMGUI_COL_WINDOW_BG: (0.1, 0.12, 0.16, 1.0),
            IMGUI_COL_BUTTON: (0.27, 0.44, 0.85, 1.0),
            IMGUI_COL_BUTTON_HOVERED: (0.36, 0.53, 0.92, 1.0),
        },
        "light": {
            IMGUI_COL_WINDOW_BG: (0.95, 0.96, 1.0, 1.0),
            IMGUI_COL_BUTTON: (0.60, 0.78, 1.0, 1.0),
            IMGUI_COL_BUTTON_HOVERED: (0.47, 0.66, 0.94, 1.0),
        },
        "retro": {
            IMGUI_COL_WINDOW_BG: (0.12, 0.10, 0.08, 1.0),
            IMGUI_COL_BUTTON: (0.93, 0.74, 0.28, 1.0),
            IMGUI_COL_BUTTON_HOVERED: (0.98, 0.84, 0.39, 1.0),
        },
    }

    def apply_theme(name: str) -> None:
        colors = themes[name]
        for col_idx, value in colors.items():
            window.imgui_set_style_color_vec4(col_idx, value)

    apply_theme(state["theme"])
    window.imgui_scale_all_sizes(2.0)
    window.imgui_set_font_global_scale(state["font_scale"])

    try:
        while window.isOpen():
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            total_time += dt

            frame_times.append(dt)
            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

            if state["animate"]:
                sample = math.sin(total_time * state["wave_speed"]) * state["wave_scale"]
                sample_history.append(sample)
            else:
                # Keep the history flat when paused so the plot stays visible.
                if sample_history:
                    sample_history.append(sample_history[-1])
                else:
                    sample_history.append(0.0)

            # Keep only the user-requested number of samples from the history.
            history_count = max(2, min(state["sample_count"], len(sample_history)))
            history_array = np.array(list(sample_history)[-history_count:], dtype=np.float32)

            width, height = window.size

            if window.imgui_begin_main_menu_bar():
                if window.imgui_begin_menu("Theme"):
                    for name in themes:
                        if window.imgui_menu_item(
                            name.title(), shortcut=None, selected=state["theme"] == name, enabled=True
                        ):
                            state["theme"] = name
                            apply_theme(name)
                    window.imgui_end_menu()

                if window.imgui_begin_menu("View"):
                    if window.imgui_menu_item(
                        "Toggle animation", shortcut="A", selected=state["animate"], enabled=True
                    ):
                        state["animate"] = not state["animate"]
                    if window.imgui_menu_item(
                        "Show plot", shortcut="P", selected=state["show_plot"], enabled=True
                    ):
                        state["show_plot"] = not state["show_plot"]
                    if window.imgui_menu_item(
                        "Reset font scale", shortcut="Ctrl+0", selected=False, enabled=True
                    ):
                        state["font_scale"] = 2.0
                        window.imgui_set_font_global_scale(state["font_scale"])
                    window.imgui_end_menu()

                if window.imgui_begin_menu("Help"):
                    if window.imgui_menu_item("About TensorFrost", shortcut=None, selected=False, enabled=True):
                        window.imgui_open_popup("about_popup")
                    window.imgui_end_menu()

                window.imgui_end_main_menu_bar()

            about_visible, _ = window.imgui_begin_popup_modal("about_popup", open=None, flags=0)
            if about_visible:
                window.imgui_text_wrapped(
                    "TensorFrost ImGui showcase demonstrating the Python bindings for the Vulkan backend."
                )
                window.imgui_spacing()
                window.imgui_text("Bindings exercised:")
                window.imgui_indent(12.0)
                window.imgui_bullet_text("Main menu bar helpers")
                window.imgui_bullet_text("Layout, widgets, and style stack APIs")
                window.imgui_bullet_text("Background draw list utilities")
                window.imgui_unindent(12.0)
                window.imgui_spacing()
                if window.imgui_button("Close##about"):
                    window.imgui_close_current_popup()
                window.imgui_end_popup()

            visible, _ = window.imgui_begin(
                "Control Center", open=None, flags=IMGUI_WINDOW_FLAGS_MENU_BAR
            )
            if visible:
                if window.imgui_begin_menu_bar():
                    if window.imgui_begin_menu("View"):
                        if window.imgui_menu_item("Reset font scale", shortcut="Ctrl+0", selected=False, enabled=True):
                            state["font_scale"] = 2.0
                            window.imgui_set_font_global_scale(state["font_scale"])
                        if window.imgui_menu_item("Show plot", shortcut="P", selected=state["show_plot"], enabled=True):
                            state["show_plot"] = not state["show_plot"]
                        window.imgui_end_menu()

                    window.imgui_end_menu_bar()

                window.imgui_text(f"Frame time: {dt * 1000.0:5.2f} ms")
                window.imgui_text_colored((0.25, 0.85, 0.45, 1.0), f"Average FPS: {fps:5.1f}")
                window.imgui_separator()

                controls_visible = window.imgui_begin_child("controls", (320, 280), border=True, flags=0)
                if controls_visible:
                    edited, new_text = window.imgui_input_text("Greeting", state["greeting"], buffer_length=128)
                    if edited:
                        state["greeting"] = new_text

                    state["animate"] = window.imgui_checkbox("Animate", state["animate"])
                    window.imgui_same_line()
                    state["show_plot"] = window.imgui_checkbox("Show plot", state["show_plot"])

                    state["wave_speed"] = window.imgui_slider_float("Wave speed", state["wave_speed"], 0.1, 5.0)
                    state["wave_scale"] = window.imgui_slider_float("Wave scale", state["wave_scale"], 0.1, 3.0)
                    state["sample_count"] = window.imgui_slider_int("History samples", state["sample_count"], 20, sample_history.maxlen)

                    window.imgui_spacing()
                    state["font_scale"] = window.imgui_slider_float("Font scale", state["font_scale"], 0.5, 2.0)
                    window.imgui_set_font_global_scale(state["font_scale"])

                    window.imgui_spacing()
                    changed, new_color = window.imgui_color_edit4("Accent color", state["accent"])
                    if changed:
                        state["accent"] = tuple(new_color)

                    window.imgui_spacing()
                    window.imgui_push_style_var_float(IMGUI_STYLEVAR_FRAME_ROUNDING, 8.0)
                    window.imgui_push_style_color(IMGUI_COL_BUTTON, state["accent"])
                    window.imgui_push_style_color(IMGUI_COL_BUTTON_HOVERED, (
                        min(1.0, state["accent"][0] + 0.1),
                        min(1.0, state["accent"][1] + 0.1),
                        min(1.0, state["accent"][2] + 0.1),
                        state["accent"][3],
                    ))
                    if window.imgui_button("Take snapshot"):
                        window.imgui_open_popup("snapshot_popup")
                    window.imgui_pop_style_color(2)
                    window.imgui_pop_style_var()

                    visible_popup, _ = window.imgui_begin_popup_modal("snapshot_popup", open=None, flags=0)
                    if visible_popup:
                        window.imgui_text_wrapped(
                            "Pretend we stored the latest plot sample to disk."
                        )
                        window.imgui_spacing()
                        if window.imgui_button("Close"):
                            window.imgui_close_current_popup()
                        window.imgui_end_popup()
                window.imgui_end_child()

                window.imgui_same_line()

                details_visible = window.imgui_begin_child("details", (0, 280), border=True, flags=0)
                if details_visible:
                    window.imgui_text("Details")
                    window.imgui_separator()
                    window.imgui_text_wrapped(
                        "The live sine wave demonstrates sliders, checkboxes, plot widgets, "
                        "menus, popups, style stacks, and color editing exposed through the Vulkan backend."
                    )

                    window.imgui_spacing()
                    window.imgui_indent(10.0)
                    window.imgui_bullet_text("Toggle themes from the menu bar.")
                    window.imgui_bullet_text("Drag the sliders to shape the waveform.")
                    window.imgui_bullet_text("Use the accent color to restyle the snapshot button.")
                    window.imgui_unindent(10.0)

                    window.imgui_spacing()
                    window.imgui_text_colored(state["accent"], f"Greeting: {state['greeting']}")

                    window.imgui_spacing()
                    window.imgui_push_style_var_vec2(IMGUI_STYLEVAR_WINDOW_PADDING, (12.0, 12.0))
                    window.imgui_text("Background text is rendered via the overlay draw list.")
                    window.imgui_pop_style_var()
                window.imgui_end_child()

                window.imgui_separator()

                if state["show_plot"] and history_array.size > 1:
                    window.imgui_plot_lines(
                        "Sine wave", history_array, values_offset=0, overlay_text="Normalized", scale_min=-3.0,
                        scale_max=3.0, graph_size=(0.0, 140.0), stride=4
                    )

                window.imgui_spacing()
                window.imgui_text(f"Window size: {width} x {height}")

                window.imgui_end()

            window.imgui_add_background_text(
                state["greeting"],
                pos=(24.0, 24.0),
                color=(state["accent"][0], state["accent"][1], state["accent"][2], 0.12),
            )

            window.present()
    finally:
        window.close()


if __name__ == "__main__":
    main()
