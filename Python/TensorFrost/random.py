from . import TensorFrost as tf

def randn2(shape, seed=0):
    #Box-Muller transform
    r1 = tf.random_value(shape, seed=seed)
    r2 = tf.random_value(shape, seed=tf.hash(seed))
    rho = tf.sqrt(-2.0*tf.log(tf.max(1e-6, r1)))
    theta = 2.0*tf.pi*r2
    return rho*tf.cos(theta), rho*tf.sin(theta)

def randn(shape, seed=0):
    return randn2(shape, seed=seed)[0]

def rand(shape, seed=0):
    return tf.random_value(shape, seed=seed)

def randn_like(tensor, seed=0):
    return randn(tensor.shape, seed=seed)

def rand_like(tensor, seed=0):
    return rand(tensor.shape, seed=seed)

def rand_int(seed, max_value):
    return tf.int(tf.pcg(tf.uint(seed)) % tf.uint(max_value))

def xor_swap(idx, n, seed):
    xor_seed = rand_int(seed, n)
    xor_idx = (idx ^ xor_seed)
    max_idx = tf.max(idx, xor_idx)
    min_idx = tf.min(idx, xor_idx)
    swap = rand_int(min_idx * 451 + seed, 2) == 0
    return tf.select(swap & (max_idx < n), xor_idx, idx)

def reverse(idx, n):
    return n - 1 - idx

def shuffle(idx, n, seed = 0, iters = 16):
    for i in range(iters):
        idx = xor_swap(idx, n, seed + i)
        idx = reverse(idx, n)
    return idx

def permutation(n, seed = 0):
    idx = tf.indices([n])[0]
    return shuffle(idx, n, seed)