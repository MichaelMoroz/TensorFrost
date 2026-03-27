import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt
import os
current_folder = os.path.dirname(os.path.abspath(__file__))

tf.initialize(tf.opengl)

def get_except_axis(a, axis):
    indices = list(a)
    indices.pop(axis) # Remove the axis dimension from the indices
    return tuple(indices)

def get_with_axis(indices, axis_index, axis):
    # Create a new tuple of indices with the axis index inserted at the specified axis
    new_indices = list(indices)
    new_indices.insert(axis, axis_index)
    return tuple(new_indices)

def intlog2(x):
    # Calculate the integer logarithm base 2 of x
    if x < 1:
        raise ValueError("x must be a positive integer")
    log = 0
    while x > 1:
        x >>= 1
        log += 1
    return log

def inplace_fft(tensor, axis = -1, inverse = False):
    shape = tensor.shape[:-1] #shape without the last dimension (complex number)
    N = shape[axis].try_get_constant()
    print("N:", N)
    if N == None or N & (N - 1) != 0:
        raise ValueError("FFT only supports constant power of 2 sizes")
    BK = min(256, N // 2) #Group size
    RADIX2 = intlog2(N)

    def expi(angle):
        return tf.cos(angle), tf.sin(angle)

    def cmul(a, b):
        return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]

    def radix2(temp, span, index, inverse):
        group_half_mask = span - 1
        group_offset = index & group_half_mask
        group_index = index - group_offset
        k1 = (group_index << 1) + group_offset
        k2 = k1 + span

        d = 1.0 if inverse else -1.0
        angle = 2 * np.pi * d * tf.float(group_offset) / tf.float(span << 1)

        #radix2 butterfly
        v1 = temp[2*k1], temp[2*k1 + 1]
        v2 = cmul(expi(angle), (temp[2*k2], temp[2*k2 + 1]))
        temp[2*k1] = v1[0] + v2[0]
        temp[2*k1 + 1] = v1[1] + v2[1]
        temp[2*k2] = v1[0] - v2[0]
        temp[2*k2 + 1] = v1[1] - v2[1]

    #workgroup mapped to our axis
    new_shape = get_except_axis(shape, axis) + (BK,)
    with tf.kernel(list(new_shape), group_size=[BK]) as indices:
        temp = tf.group_buffer(N*2, tf.float32)
        if not isinstance(indices, tuple):
            indices = (indices,)
        
        tx = indices[0].block_thread_index(0)
        indices = get_except_axis(indices, -1) #skip the workgroup axis
        M = N // BK

        for i in range(M):
            rowIndex = i * BK + tx
            idx = tf.int(tf.reversebits(tf.uint(rowIndex)) >> (32 - RADIX2))
            tensor_index = get_with_axis(indices, rowIndex, axis)
            temp[2*idx] = tensor[tensor_index + (0,)]
            temp[2*idx + 1] = tensor[tensor_index + (1,)]

        tf.group_barrier()

        span = 1
        while span < N:
            for j in range(M // 2):
                rowIndex = j * BK + tx
                radix2(temp, span, rowIndex, inverse)
            tf.group_barrier()
            span *= 2

        for i in range(M):
            rowIndex = i * BK + tx
            factor = 1.0 / float(N if inverse else 1)
            tensor_index = get_with_axis(indices, rowIndex, axis)
            tensor[tensor_index + (0,)] = temp[2*rowIndex] * factor
            tensor[tensor_index + (1,)] = temp[2*rowIndex + 1] * factor

target_res = 1024

def fft():
    A = tf.input([target_res, target_res, 2], tf.float32)
    B = tf.copy(A)
    inplace_fft(B, axis=0, inverse=False)
    inplace_fft(B, axis=1, inverse=False)
    return B

def ifft():
    A = tf.input([target_res, target_res, 2], tf.float32)
    B = tf.copy(A)
    inplace_fft(B, axis=0, inverse=True)
    inplace_fft(B, axis=1, inverse=True)
    return B

fft = tf.compile(fft)
ifft = tf.compile(ifft)

all_kernels = tf.get_all_generated_kernels()
print("Generated kernels:")
for k in all_kernels:
    print(k[0][2])

input_img = np.array(plt.imread(current_folder+"/test.png"), dtype=np.float32)
image_resampled = np.pad(input_img, ((0, target_res - input_img.shape[0]), (0, target_res - input_img.shape[1]), (0, 0)), 'constant')

# plt.imshow(image_resampled)
# plt.show()
# print(image_resampled.shape)

r_channel = image_resampled[..., 0]
complex_image = np.zeros((target_res, target_res, 2), dtype=np.float32)
complex_image[..., 0] = r_channel
complex_image[..., 1] = np.zeros((target_res, target_res), dtype=np.float32)

image_tf = tf.tensor(complex_image)
transformed_tf = fft(image_tf)
transformed = transformed_tf.numpy

#plot the magnitude of the transformed image
plt.imshow(np.log(np.abs(transformed[..., 0] + 1j * transformed[..., 1])))
plt.colorbar()
plt.title("FFT")
plt.show()

image_recovered_tf = ifft(transformed_tf)
image_recovered = image_recovered_tf.numpy
#plot the recovered image
plt.imshow(image_recovered[..., 0])
plt.colorbar()
plt.title("Recovered Image")
plt.show()