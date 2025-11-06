from pathlib import Path
import math

import numpy as np
import TensorFrost as tf


def load_shader() -> str:
    with open(Path(__file__).with_name("mandelbrot.slang"), "r", encoding="utf-8") as handle:
        return handle.read()


def main() -> None:
    width, height = 1024, 768
    win = tf.createWindow(width, height, "Mandelbrot (ImGui)")

    fmt = int(win.format)
    is_bgra_default = fmt in (44, 50)  # VK_FORMAT_B8G8R8A8_UNORM / _SRGB

    pixel_capacity = max(1, width * height)
    pixel_buffer = tf.createBuffer(pixel_capacity, 4, False)
    params_buffer = tf.createBuffer(8, 4, True)

    shader_source = load_shader()
    program = tf.createComputeProgramFromSlang("mandelbrot", shader_source, "csMain", ro_count=1, rw_count=1)

    center = [-0.5, 0.0]
    scale = 3.0
    log_scale = math.log10(scale)
    manual_iterations = 500
    auto_iterations = True
    swap_rb = is_bgra_default
    plot_history = np.zeros(120, dtype=np.float32)
    history_index = 0

    params = np.zeros(8, dtype=np.float32)

    def ensure_pixel_buffer(cur_width: int, cur_height: int) -> None:
        nonlocal pixel_buffer, pixel_capacity
        required = max(1, cur_width * cur_height)
        if required != pixel_capacity:
            pixel_buffer.release()
            pixel_buffer = tf.createBuffer(required, 4, False)
            pixel_capacity = required

    try:
        while win.isOpen():
            width, height = win.size
            width = max(1, int(width))
            height = max(1, int(height))

            ensure_pixel_buffer(width, height)

            aspect = height / float(width)

            visible, _ = win.imgui_begin("Mandelbrot Controls", open=True)
            if visible:
                win.imgui_text(f"Resolution: {width} × {height}")
                win.imgui_text(f"Format: {fmt}")
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
            xmin = center[0] - xspan * 0.5
            ymin = center[1] - yspan * 0.5
            dx = xspan / width
            dy = yspan / height

            params[0] = float(width)
            params[1] = float(height)
            params[2] = xmin
            params[3] = ymin
            params[4] = dx
            params[5] = dy
            params[6] = float(manual_iterations)
            params[7] = 1.0 if swap_rb else 0.0

            params_buffer.setData(params)
            program.run([params_buffer], [pixel_buffer], width * height)

            win.drawBuffer(pixel_buffer, width, height)
    finally:
        win.close()
        pixel_buffer.release()
        params_buffer.release()


if __name__ == "__main__":
    main()
