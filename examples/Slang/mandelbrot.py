from pathlib import Path
import math
import time

import numpy as np
import TensorFrost as tf


def load_shader() -> str:
    with open(Path(__file__).with_name("mandelbrot.slang"), "r", encoding="utf-8") as handle:
        return handle.read()


def main() -> None:
    width, height = 1024, 768
    win = tf.createWindow(width, height, "Mandelbrot (ImGui)")
    win.imgui_scale_all_sizes(2.0)
    win.imgui_set_font_global_scale(2.0)

    fmt = int(win.format)
    is_bgra_default = fmt in (44, 50)  # VK_FORMAT_B8G8R8A8_UNORM / _SRGB

    pixel_capacity = max(1, width * height)
    pixel_buffer = tf.createBuffer(pixel_capacity, 4, False)

    shader_source = load_shader()
    program = tf.createComputeProgramFromSlang(
        "mandelbrot",
        shader_source,
        "csMain",
        ro_count=0,
        rw_count=1,
        push_constant_size=32,
    )
    local_size = 64

    center = [-0.5, 0.0]
    scale = 3.0
    log_scale = math.log10(scale)
    pending_scroll = 0.0
    manual_iterations = 500
    auto_iterations = True
    swap_rb = is_bgra_default
    plot_history = np.zeros(120, dtype=np.float32)
    history_index = 0

    params = np.zeros(8, dtype=np.float32)
    prev_mouse_pos = win.mouse_position()
    dragging = False
    prev_time = time.perf_counter()
    fps = 0.0

    def ensure_pixel_buffer(cur_width: int, cur_height: int) -> None:
        nonlocal pixel_buffer, pixel_capacity
        required = max(1, cur_width * cur_height)
        if required != pixel_capacity:
            pixel_buffer = tf.createBuffer(required, 4, False)
            pixel_capacity = required

    while win.isOpen():
        now = time.perf_counter()
        dt = max(now - prev_time, 1e-6)
        prev_time = now
        fps = fps * 0.9 + (1.0 / dt) * 0.1

        width, height = win.size
        width = max(1, int(width))
        height = max(1, int(height))
        thread_count = max(1, width * height)
        group_count = max((thread_count + local_size - 1) // local_size, 1)

        ensure_pixel_buffer(width, height)

        if pending_scroll:
            scroll_adjust = pending_scroll * 0.12
            log_scale = max(min(log_scale - scroll_adjust, 0.5), -4.5)
            scale = pow(10.0, log_scale)
            pending_scroll = 0.0

        aspect = height / float(width)

        visible, _ = win.imgui_begin("Mandelbrot Controls", open=True)
        if visible:
            win.imgui_text(f"Resolution: {width} × {height}")
            win.imgui_text(f"Format: {fmt}")
            win.imgui_text(f"FPS: {fps:5.1f} | {dt * 1000.0:.2f} ms")
            log_scale = win.imgui_slider_float("log₁₀ scale", log_scale, -4.5, 0.5)
            scale = pow(10.0, log_scale)
            center[0] = win.imgui_slider_float("Center X", center[0], -2.5, 1.5)
            center[1] = win.imgui_slider_float("Center Y", center[1], -1.5, 1.5)
            auto_iterations = win.imgui_checkbox("Auto iterations", auto_iterations)
            if auto_iterations:
                auto_value = max(64, int(200 + (-math.log10(scale)) * 120))
                manual_iterations = auto_value
                win.imgui_text(f"Max iterations (auto): {auto_value}")
            else:
                manual_iterations = win.imgui_slider_int("Max iterations", manual_iterations, 64, 5000)
            swap_rb = win.imgui_checkbox("BGRA swap", swap_rb)
            if win.imgui_button("Reset view"):
                center[:] = (-0.5, 0.0)
                scale = 3.0
                log_scale = math.log10(scale)
                manual_iterations = 500
                auto_iterations = True
                pending_scroll = 0.0

            plot_history[history_index % plot_history.size] = float(manual_iterations)
            history_index += 1
            win.imgui_plot_lines(
                "Iteration history",
                plot_history,
                values_offset=history_index % plot_history.size,
                graph_size=(0.0, 60.0),
            )
        win.imgui_end()

        xspan = scale
        yspan = xspan * aspect
        dx = xspan / width
        dy = yspan / height

        mouse_pos = win.mouse_position()
        want_capture_mouse = win.imgui_want_capture_mouse()
        mouse_down = win.is_mouse_button_pressed(0)
        dragging_now = mouse_down and not want_capture_mouse
        if dragging and dragging_now:
            delta_x = mouse_pos[0] - prev_mouse_pos[0]
            delta_y = mouse_pos[1] - prev_mouse_pos[1]
            center[0] -= delta_x * dx
            center[1] -= delta_y * dy
        dragging = dragging_now
        prev_mouse_pos = mouse_pos

        xmin = center[0] - xspan * 0.5
        ymin = center[1] - yspan * 0.5
        params[0] = float(width)
        params[1] = float(height)
        params[2] = xmin
        params[3] = ymin
        params[4] = dx
        params[5] = dy
        params[6] = float(manual_iterations)
        params[7] = 1.0 if swap_rb else 0.0

        program.run([], [pixel_buffer], group_count, params)

        win.drawBuffer(pixel_buffer, width, height)

        _, scroll_dy = win.consume_scroll_delta()
        if not want_capture_mouse:
            pending_scroll += scroll_dy


if __name__ == "__main__":
    main()
