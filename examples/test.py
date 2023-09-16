import TensorFrost as tf
import numpy as np

def ParticleExample(inputs):
    pos = inputs[0]
    vel = inputs[1]

    shape = pos.shape
    indices = tf.indices([shape[0], shape[0]])
    p1 = indices[0]
    p2 = indices[1]

    # Calculate the distance between each particle
    #x1 = pos[p1, 0]
    #y1 = pos[p1, 1]
#
    #x2 = pos[p2, 0]
    #y2 = pos[p2, 1]
#
    #dx = x1 - x2
    #dy = y1 - y2
#
    #eps = 1e-3
    #dist2 = dx**2 + dy**2
    #dist1 = tf.sqrt(dist2 + eps**2)
#
    ## Calculate the potential energy between each particle
    #Vi = 1.0 / dist1
#
    ## Calculate the force between each particle
    #Fx = -Vi.grad(x1)
    #Fy = -Vi.grad(y1)
#
    ##F = tf.tensor(shape, 0.0)
    ##tf.scatterAdd(F[p1, 0], Fx)
    ##tf.scatterAdd(F[p1, 1], Fy)
#
    #Fx = Fx.sum(dim=1)
    #Fy = Fy.sum(dim=1)
    #F = tf.stack([Fx, Fy], axis=1)

    # Calculate the distance between each particle
    pos1 = pos[p1, :]
    pos2 = pos[p2, :]
    dist = pos1 - pos2
    eps = 1e-3
    dist2 = tf.sum(dist**2, axis=2)
    dist1 = tf.sqrt(dist2 + eps**2)

    # Calculate the potential energy between each particle
    Vi = 1.0 / dist1

    # Calculate the force between each particle
    F = -Vi.grad(pos1)
    F = F.sum(dim=1)

    # Integrate the velocity and position
    dt = 0.1
    vel += F * dt
    pos += vel * dt

    return [pos, vel]

def Jacobi(field, source):
    indices = field.indices
    i = indices[0]
    j = indices[1]

    p_l = field[i - 1, j]
    p_r = field[i + 1, j]
    p_t = field[i, j - 1]
    p_b = field[i, j + 1]

    return (p_l + p_r + p_t + p_b + source) / 4.0

def JacobiSolver(inputs):
    field = inputs[0]
    source = inputs[1]
    iterations = inputs[2]

    def loopbody(iteration):
        field = Jacobi(field, source)

    tf.loop(iterations, loopbody)

    return [field]

def sdSphere(position, center, radius):
    return tf.length(position - center) - radius

def sdBox(position, center, size):
    d = tf.abs(position - center) - size
    return tf.length(tf.max(d, 0.0)) + tf.min(tf.max(d.x, tf.max(d.y, d.z)), 0.0)

def sdPlane(position, normal, distance):
    return tf.dot(position, normal) + distance

def sdScene(position):
    sphere = sdSphere(position, tf.mat([0.0, 0.0, 0.0]), 1.0)
    box = sdBox(position, tf.mat([1.0, 0.0, 0.0]), tf.mat([1.0, 1.0, 1.0]))
    plane = sdPlane(position, tf.mat([0.0, 1.0, 0.0]), 0.0)

    return tf.min(sphere, tf.min(box, plane))

def RayMarch(inputs):
    position = inputs[0]
    direction = inputs[1]

    minDistance = 0.01
    maxIterations = 100

    cur_pos = position
    td = 0.0

    def loopbody(iteration):
        distance = sdScene(cur_pos)
        td += distance
        cur_pos += direction * distance

        tf.cond(distance < minDistance, lambda: [tf.loopbreak()], lambda: [])

    tf.loop(maxIterations, loopbody)

    return [td]

# Simple neural network example
def NeuralNetwork(input, weights, biases):
    layer = tf.matmul(input, weights[0]) + biases[0]
    layer = tf.relu(layer)
    layer = tf.matmul(layer, weights[1]) + biases[1]
    return layer

# Initializing the weights and biases
def InitWeights(shape):
    return tf.random(shape, -1.0, 1.0)

def InitNeuralNetwork():
    weights = [InitWeights([2, 3]), InitWeights([3, 1])]
    biases = [InitWeights([1, 3]), InitWeights([1, 1])]
    return [weights, biases]

def NeuralNetworkOptimizationIteration(inputs):
    weights = inputs[0]
    biases = inputs[1]

    # Create a random input and output
    input = tf.random([1, 2], -1.0, 1.0)
    output = tf.random([1, 1], -1.0, 1.0)

    # Create a neural network
    prediction = NeuralNetwork(input, weights, biases)

    # Calculate the loss
    loss = tf.sqr(prediction - output)

    # go over the weights and biases and calculate the gradients and update the weights using gradient descent
    for i in range(len(weights)):
        weights[i] -= loss.grad(weights[i]) * 0.01
        biases[i] -= loss.grad(biases[i]) * 0.01

    return [weights, biases]

def modified_gram_schmidt(A):
    """
    Implements the Modified Gram-Schmidt orthogonalization to get the QR decomposition of matrix A.
    A = QR
    """
    A = A.astype(float)  # Ensure A is of float type
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for i in range(n):
        R[i, i] = np.linalg.norm(A[:, i])
        Q[:, i] = A[:, i] / R[i, i]
        R[i, i+1:n] = np.dot(Q[:, i].T, A[:, i+1:n])
        A[:, i+1:n] -= np.outer(Q[:, i], R[i, i+1:n])
    
    return Q, R

def QRDecomposition(inputs):
    A = inputs[0]
    m, n = A.shape
    Q = tf.zeros([m, n])
    R = tf.zeros([n, n])

    def loop(i):
        R[i, i] = tf.norm(A[:, i])
        Q[:, i] = A[:, i] / R[i, i]

        j = tf.range(i + 1, n)
        R[i, j] = tf.dot(Q[:, i].T, A[:, j])
        A[:, j] -= tf.outer(Q[:, i], R[i, j])
       
    tf.loop(n, loop)

    return [Q, R]

def InitWaveSimulation(inputs):
    Xdim = 2048
    Ydim = 2048

    ids = tf.indices([Xdim, Ydim])
    i = ids[0]
    j = ids[1]

    x = tf.f32(i) / Xdim - 0.5
    y = tf.f32(j) / Ydim - 0.5

    u = tf.exp(-10000.0 * (x * x + y * y))
    v = tf.zeros([Xdim, Ydim])

    return [u, v]

def StepWaveSimulation(inputs):
    u = inputs[0]
    v = inputs[1]

    Xdim = u.shape[0]
    Ydim = u.shape[1]

    ids = tf.indices([Xdim, Ydim])
    i = ids[0]
    j = ids[1]

    u_laplacian = u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - u * 4.0

    #boundary conditions
    edge = 1.0 - tf.f32((i == 0) | (i == Xdim - 1) | (j == 0) | (j == Ydim - 1))

    #verlet integration
    dt = 0.5
    v_new = (v + u_laplacian * dt) * edge
    u_new = (u + v_new * dt) * edge

    return [u_new, v_new]

tf.backend("cpu")

particleProgram = tf.CompileProgram(ParticleExample)
jacobiSolverProgram = tf.CompileProgram(JacobiSolver)
rayMarchProgram = tf.CompileProgram(RayMarch)
weightInitProgram = tf.CompileProgram(InitNeuralNetwork)
neuralNetworkOptimizationIterationProgram = tf.CompileProgram(NeuralNetworkOptimizationIteration)

#get the graphs 
graph1 = particleProgram.debugExecutionGraph
graph2 = particleProgram.debugTensorGraph

#get the program json
json = particleProgram.json

# initialize some particles (by using numpy)
pos = np.random.rand(100, 2)
vel = np.random.rand(100, 2)

# create a tensor from the numpy arrays
pos = tf.numpy(pos)
vel = tf.numpy(vel)

# execute the program
pos, vel = particleProgram([pos, vel])

# Create a 2x3 tensor
tensor = [[4, 5], [6, 7]] * tf.tensor([2, 2]) 

print(tensor.shape)
print(tensor.type)
