import TensorFrost as tf
import numpy as np
import time

tf.initialize(tf.opengl)

S = 1024
MAX_ITER = 3000
SAMPLES = 4

T1 = MAX_ITER // 50
T2 = MAX_ITER // 15
T3 = MAX_ITER

def buddhabrot():
    prev_frame = tf.input([S, S, 3], tf.float32)
    frame_id = tf.input([1], tf.int32)
    time_var = tf.input([1], tf.float32)[0]

    atomic_canvas = tf.zeros([S, S, 3], tf.int32)
    i, j = tf.indices([S, S])
    x, y = tf.float(i), tf.float(j)
    seed = tf.uint(i + j * S + frame_id[0] * S * S)
    aspect = tf.float(S) / tf.float(S)

    def rand():
        seed.val = tf.pcg(seed)
        return tf.float(seed) / tf.float(0xFFFFFFFF)
    
    def randn():
        # Box-Muller transform
        u1, u2 = rand(), rand()
        r1 = tf.sqrt(-2.0 * tf.log(u1))
        theta = 2.0 * np.pi * u2
        return r1 * tf.cos(theta), r1 * tf.sin(theta)

    with tf.loop(SAMPLES):
        cx, cy = randn()
        z0x = tf.const(0.0)
        z0y = tf.const(0.0)
        z_re = tf.copy(z0x)
        z_im = tf.copy(z0y)
        l = tf.zeros([], tf.int32)
        c_re = cx
        c_im = cy

        def mandelbrot_iter(z_re, z_im):
            z_re_new = z_re*z_re - z_im*z_im + c_re
            z_im_new = 2.0*z_re*z_im + c_im
            z_re.val = z_re_new
            z_im.val = z_im_new
            with tf.if_cond((z_re*z_re + z_im*z_im) > 8.0):
                tf.break_loop()

        with tf.loop(MAX_ITER):
            mandelbrot_iter(z_re, z_im)
            l.val += 1

        with tf.if_cond(l >= MAX_ITER):
            tf.continue_loop()

        z_re.val = z0x
        z_im.val = z0y

        with tf.loop(MAX_ITER):
            mandelbrot_iter(z_re, z_im)
            
            px = (tf.cos(.3*time_var) * z_re + tf.sin(.3*time_var) * c_re) / 1.5 / aspect * 0.5 + 0.6
            py = (tf.cos(.3*time_var) * z_im + tf.sin(.3*time_var) * c_im) / 1.5 / aspect * 0.5 + 0.5
            with tf.if_cond((px < 0.0) | (px > 1.0) | (py < 0.0) | (py > 1.0)):
                tf.continue_loop()
            x_pix = tf.int(px * tf.float(S))
            y_pix = tf.int(py * tf.float(S))
            
            ch = tf.select(l < T1, 2, tf.select(l < T2, 1, 0))
            tf.scatterAdd(atomic_canvas[S - 1 - y_pix, x_pix, ch], 1)
            tf.scatterAdd(atomic_canvas[y_pix, x_pix, ch], 1)

    canvas = tf.buffer([S, S, 3], tf.float32)

    x = tf.float(atomic_canvas[i, j, 0])
    y = tf.float(atomic_canvas[i, j, 1])
    z = tf.float(atomic_canvas[i, j, 2])
    rx = x + y + z
    ry = y + z
    rz = z

    norm = float(MAX_ITER * SAMPLES) / 20.0

    fid =  tf.float(frame_id[0])
    k = 0.6
    canvas[i, j, 0] = tf.lerp(tf.smoothstep(0.0, 1.0, 2.5 * (rx/norm) ** 0.9), prev_frame[i, j, 0], k)
    canvas[i, j, 1] = tf.lerp(tf.smoothstep(0.0, 1.0, 2.5 * (ry/norm) ** 0.65), prev_frame[i, j, 1], k)
    canvas[i, j, 2] = tf.lerp(tf.smoothstep(0.0, 1.0, 2.5 * (rz/norm) ** 0.5), prev_frame[i, j, 2], k)

    return [canvas, frame_id + 1]

mand = tf.compile(buddhabrot)

frame = np.zeros([S, S, 3], np.float32)
frame_tf = tf.tensor(frame)
frame_id = tf.tensor(np.array([0], np.int32))

tf.show_window(S, S, "Buddhabrot")

init_time = time.time()

while not tf.window_should_close():
    cur_time = time.time() - init_time
    time_tf = tf.tensor(np.array([cur_time], np.float32))
    frame_tf, frame_id = mand(frame_tf, frame_id, time_tf)
    tf.render_frame(frame_tf)
    render_time = time.time() - init_time - cur_time
    tf.imgui_text("Render time: %.3f ms" % (render_time * 1000))

tf.hide_window()