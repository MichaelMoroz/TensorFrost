import numpy as np
import TensorFrost as tf
from pathlib import Path

ctx = tf.VulkanContext()

W, H = 1024, 768
win = tf.createWindow(ctx, W, H, "Mandelbrot")
fmt = int(win.format)
is_bgra = fmt in (44, 50)  # VK_FORMAT_B8G8R8A8_UNORM / _SRGB

pix = tf.createBuffer(ctx, W*H, 4, False)        # uint32 pixels
params = tf.createBuffer(ctx, 8, 4, True)        # 8 float32 params

with open(Path(__file__).with_name('mandelbrot.slang'), 'r') as f:
    hlsl = f.read()

prog = tf.createComputeProgramFromSlang(ctx, "mandelbrot", hlsl, "csMain", roCount=1, rwCount=1)

# view rectangle with aspect correction
xspan = 3.0
yspan = xspan * (H / float(W))
xmin, ymin = -2.0, -yspan * 0.5
dx, dy = xspan / W, yspan / H
max_iter = 500.0

p = np.array([float(W), float(H), xmin, ymin, dx, dy, max_iter, 1.0 if is_bgra else 0.0], dtype=np.float32)
tf.setBufferData(ctx, params, p)

try:
    while tf.windowOpen(win):
        tf.runProgram(ctx, prog, [params], [pix], W*H)
        tf.drawBuffer(win, pix, W, H)
finally:
    tf.destroyComputeProgram(ctx, prog)
    tf.destroyBuffer(ctx, pix)
    tf.destroyBuffer(ctx, params)
