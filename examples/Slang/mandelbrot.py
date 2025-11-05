import numpy as np
import TensorFrost as tf
from pathlib import Path

W, H = 1024, 768
win = tf.createWindow(W, H, "Mandelbrot")
fmt = int(win.format)
is_bgra = fmt in (44, 50)  # VK_FORMAT_B8G8R8A8_UNORM / _SRGB

pix = tf.createBuffer(W*H, 4, False)        # uint32 pixels
params = tf.createBuffer(8, 4, True)        # 8 float32 params

with open(Path(__file__).with_name('mandelbrot.slang'), 'r') as f:
    hlsl = f.read()

prog = tf.createComputeProgramFromSlang("mandelbrot", hlsl, "csMain", ro_count=1, rw_count=1)

# view rectangle with aspect correction
xspan = 3.0
yspan = xspan * (H / float(W))
xmin, ymin = -2.0, -yspan * 0.5
dx, dy = xspan / W, yspan / H
max_iter = 500.0

p = np.array([float(W), float(H), xmin, ymin, dx, dy, max_iter, 1.0 if is_bgra else 0.0], dtype=np.float32)
params.setData(p)

try:
    while win.isOpen():
        prog.run([params], [pix], W*H)
        win.drawBuffer(pix, W, H)
finally:
    win.close()
