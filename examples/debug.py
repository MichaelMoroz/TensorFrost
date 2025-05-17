import numpy as np
import TensorFrost as tf

tf.initialize(tf.cpu)

blur_d = 16
blur_r = blur_d * 0.5

def kernel(r):
    #return 1.0
    return tf.exp(-0.5 * (r / blur_r)**2.0) / (blur_r * np.sqrt(2.0 * np.pi))

# def blurfunc():
#     img = tf.func_input([-1, -1], tf.float32)
#     with tf.loop(-blur_d, blur_d+1, 1) as k:
#         blur_h += img[i+k, j] * kernel(tf.float(k))
#     with tf.loop(-blur_d, blur_d+1, 1) as k:
#         blur_v += blur_h[i, j+k] * kernel(tf.float(k))
#     return blur_v
#
# def blur():
#     img = tf.input([-1, -1, -1], tf.float32)
#     N, M, C = img.shape
#     blur_h = tf.zeros(img.shape, tf.float32)
#     blur_v = tf.zeros(img.shape, tf.float32)
#     i, j, ch = img.indices
#
#     tf.vmap(inputs=[img], map=[C], func=blurfunc);
#
#     return blur_v

@tf.compile
def blur(img: tf.Arg([-1, -1, -1], tf.float32)):
    blur_h = tf.zeros(img.shape, tf.float32)
    blur_v = tf.zeros(img.shape, tf.float32)
    i, j, ch = img.indices

    #horizontal blur
    with tf.loop(-blur_d, blur_d+1, 1) as k:
        blur_h += img[i+k, j, ch] * kernel(tf.float(k))

    #vertical blur
    with tf.loop(-blur_d, blur_d+1, 1) as k:
        blur_v += blur_h[i, j+k, ch] * kernel(tf.float(k))

    return blur_v
