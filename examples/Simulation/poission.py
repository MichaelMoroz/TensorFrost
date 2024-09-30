import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

tf.initialize(tf.opengl)

# red-black Gauss-Seidel
def RBGS(pressure0, div, iterations, overrelax=1.0, rbgs=True):
    N1, M1 = pressure0.shape

    pressure1 = tf.copy(pressure0)
    pressure2 = tf.copy(pressure0)
    
    i, j = pressure0.indices   

    def sample(pressure, i, j):
        return tf.select((i >= 0) & (i < N1) & (j >= 0) & (j < M1), pressure[i, j], 0.0)

    def stencil(pressure, i, j):
        return (sample(pressure, i-1, j) + sample(pressure, i+1, j) + sample(pressure, i, j-1) + sample(pressure, i, j+1) - div[i, j]) / 4.0

    error = tf.buffer([iterations], tf.float32)

    # pressure solve
    with tf.loop(iterations) as it:
        # red
        cur_overrelax = overrelax
        is_red = tf.select(rbgs, ((i + j) % 2 == 0), True)
        pressure2[i,j] = tf.select(is_red, tf.lerp(pressure1, stencil(pressure1, i, j), cur_overrelax), pressure1)

        # black
        is_black = tf.select(rbgs, ((i + j) % 2 == 1), True)
        pressure1[i,j] = tf.select(is_black, tf.lerp(pressure2, stencil(pressure2, i, j), cur_overrelax), pressure2)

        residue = tf.abs(sample(pressure1, i-1, j) + sample(pressure1, i+1, j) + sample(pressure1, i, j-1) + sample(pressure1, i, j+1) - 4.0*pressure1 - div)

        error[it] = tf.log(tf.mean(tf.mean(residue)))/tf.log(10.0)

    return pressure1, error


def Solver():
    field = tf.input([-1, -1], tf.float32)
    div = tf.input(field.shape, tf.float32)
    params = tf.input([-1], tf.float32)

    iters = tf.int(params[0])
    overrelax = params[1]
    rbgs = tf.bool(params[2])

    field, error = RBGS(field, div, iters, overrelax, rbgs)

    field_render = tf.abs(tf.repeat(tf.unsqueeze(field), 3))

    return field_render, error


solver = tf.compile(Solver)

size = 128

h = 1.0 / size
b = np.zeros([size, size], np.float32)
# for j in range(size):
#     for i in range(size):
#         x = (i+1) * h
#         y = (j+1) * h
#         b[j,i] = 32.0*((y-1)*y + (x-1)*x)*h*h
b[size//2, size//2] = 1.0
    
div = tf.tensor(b)
field = tf.tensor(-0.0*b/h/h)

tf.window.show(1024, 1024, "Solver")

steps = 256
overrelax = 1.954
use_rbgs = True
plot_field = True

while not tf.window.should_close():
    field1, meanres = solver(field, div, [steps, overrelax, use_rbgs])

    tf.imgui.begin("RBGS")
    steps = tf.imgui.slider("steps", steps, 1, 512)
    overrelax = tf.imgui.slider("overrelax", overrelax, 1.0, 2.0)
    use_rbgs = tf.imgui.checkbox("use_rbgs", use_rbgs)
    plot_field = tf.imgui.checkbox("plot_field", plot_field)
    tf.imgui.plotlines("residue history", meanres.numpy, graph_size=(0, 400))
    tf.imgui.end()
    
    tf.window.render_frame(field1)

#plot mean residue
plt.plot(meanres.numpy)
plt.xlabel('steps')
plt.ylabel('mean residue')
plt.title('Poisson')
plt.grid()
plt.show()

# iters = 256
# overrelax_to_test = [1.0, 1.2, 1.4, 1.6, 1.8]

# results_rbgs = {}
# for overrelax in overrelax_to_test:
#     results_rbgs[overrelax] = []
#     for i in range(iters):
#         field1, res, meanres = solver(field, div, [i, overrelax, True])
#         results_rbgs[overrelax].append(meanres.numpy[0])

# results_jacobi = {}
# for overrelax in overrelax_to_test:
#     results_jacobi[overrelax] = []
#     for i in range(iters):
#         field1, res, meanres = solver(field, div, [i, overrelax, False])
#         results_jacobi[overrelax].append(meanres.numpy[0])
        

# for overrelax in overrelax_to_test:
#     plt.plot(results_rbgs[overrelax], label='overrelax = %.2f, RBGS' % overrelax)
#     plt.plot(results_jacobi[overrelax], label='overrelax = %.2f, Jacobi' % overrelax)

# plt.yscale('log')
# plt.xlabel('steps')
# plt.ylabel('mean residue')
# plt.title('Poisson')
# plt.legend()
# plt.grid()
# plt.show()



