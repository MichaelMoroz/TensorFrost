import numpy as np
import TensorFrost as tf
import torch
import imageio
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax.numpy as jnp
from jax import jit, vmap
from jax.experimental import sparse

tf.initialize(tf.opengl)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def n_body():
    X = tf.input([-1, 3], tf.float32)
    N = X.shape[0]
    V = tf.input([N, 3], tf.float32)

    dx = tf.unsqueeze(X, axis=1) - tf.unsqueeze(X, axis=0)

    # Compute the SPH forces
    d2 = tf.unsqueeze(tf.sum(dx**2.0, axis=-1), axis=-1)
    dist = tf.sqrt(d2 + 1e-4) # soft distance
    Fg = -dx * 1.0 / (d2 + 1e-4) * 1.0 / dist
  
    Fi = tf.sum(Fg, axis=1)

    dt = 0.001

    Vnew = V + Fi * dt
    Xnew = X + Vnew * dt

    return Xnew, Vnew

def n_body_loop():
    X = tf.input([-1, 3], tf.float32)
    N = X.shape[0]
    V = tf.input([N, 3], tf.float32)

    Fi = tf.buffer([N, 3], tf.float32)

    i, = tf.indices([N])
    Fix, Fiy, Fiz = tf.const(0.0), tf.const(0.0), tf.const(0.0)
    x0, y0, z0 = X[i, 0], X[i, 1], X[i, 2]
    with tf.loop(N) as j:
        x, y, z = X[j, 0], X[j, 1], X[j, 2]
        dx, dy, dz = x - x0, y - y0, z - z0
        d2 = dx*dx + dy*dy + dz*dz
        dist = tf.sqrt(d2 + 1e-4)
        Fg = -dx / (d2 + 1e-4) * 1.0 / dist
        Fix.val += Fg * dx
        Fiy.val += Fg * dy
        Fiz.val += Fg * dz
        
    Fi[i, 0] = Fix
    Fi[i, 1] = Fiy
    Fi[i, 2] = Fiz

    dt = 0.001

    Vnew = V + Fi * dt
    Xnew = X + Vnew * dt

    return Xnew, Vnew

def n_body_torch(X, V):
    N = X.shape[0]
    dx = X.unsqueeze(1) - X.unsqueeze(0)
    
    # Compute the SPH forces
    d2 = torch.sum(dx**2.0, dim=-1).unsqueeze(-1)
    dist = torch.sqrt(d2 + 1e-4)  # soft distance
    Fg = -dx * 1.0 / (d2 + 1e-4) * 1.0 / dist
    
    Fi = torch.sum(Fg, dim=1)
    
    dt = 0.001
    
    Vnew = V + Fi * dt
    Xnew = X + Vnew * dt
    
    return Xnew, Vnew

def n_body_jax(X, V):
    N = X.shape[0]
    dx = X.reshape(N, 1, 3) - X.reshape(1, N, 3)
    
    # Compute the SPH forces
    d2 = jnp.sum(dx**2.0, axis=-1)[:, :, None]
    dist = jnp.sqrt(d2 + 1e-4)  # soft distance
    Fg = -dx * 1.0 / (d2 + 1e-4) * 1.0 / dist
    
    Fi = jnp.sum(Fg, axis=1)
    
    dt = 0.001
    
    Vnew = V + Fi * dt
    Xnew = X + Vnew * dt
    
    return Xnew, Vnew

nbody_tf = tf.compile(n_body)
nbody_tf_loop = tf.compile(n_body_loop)

n_body_torch_compiled = torch.compile(n_body_torch)
n_body_jax_compiled = jit(n_body_jax)

def performance_comparison(iters=1000, N=1000):
    X = 5.0 * np.random.randn(N, 3).astype(np.float32)
    V = np.zeros((N, 3), dtype=np.float32)

    # PyTorch setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TensorFrost timing
    X_tf = tf.tensor(X)
    V_tf = tf.tensor(V)
    start_time = time.time()
    for _ in range(iters):
        X_tf, V_tf = nbody_tf(X_tf, V_tf)
    X_np1 = X_tf.numpy 
    tf_time = time.time() - start_time

    X_tf = tf.tensor(X)
    V_tf = tf.tensor(V)
    start_time = time.time()
    for _ in range(iters):
        X_tf, V_tf = nbody_tf_loop(X_tf, V_tf)
    X_np1 = X_tf.numpy
    tf_time_loop = time.time() - start_time

    # PyTorch timing
    # X_torch = torch.tensor(X, device=device)
    # V_torch = torch.tensor(V, device=device)
    # start_time = time.time()
    # with torch.no_grad():
    #     for _ in range(iters):
    #         X_torch, V_torch = n_body_torch(X_torch, V_torch)

    # X_np2 = X_torch.cpu().numpy()
    # torch_time = time.time() - start_time

    # PyTorch compiled timing
    X_torch = torch.tensor(X, device=device)
    V_torch = torch.tensor(V, device=device)
    start_time = time.time()
    for _ in range(iters):
        X_torch, V_torch = n_body_torch_compiled(X_torch, V_torch)
    X_np2 = X_torch.cpu().numpy()
    torch_time_compiled = time.time() - start_time

    # JAX timing
    X_jax = jnp.array(X)
    V_jax = jnp.array(V)
    start_time = time.time()
    for _ in range(iters):
        X_jax, V_jax = n_body_jax_compiled(X_jax, V_jax)
    jax_time = time.time() - start_time

    return tf_time / iters, torch_time_compiled / iters, jax_time / iters, tf_time_loop / iters

#performance_comparison(1000)

# Do performance comparisons for different numbers of particles
N0 = 100
dN = 500
runs = 10
N1 = N0 + dN * runs

times = []

for N in tqdm(range(N0, N1, dN)):
    speedup = performance_comparison(4000000//N, N)
    times.append(speedup)
    #force cleanup of all GPU memory
    torch.cuda.empty_cache()

def plot_performance_comparison(times, N0, N1, runs):
    particles = np.arange(N0, N1, dN)

    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.2
    index = np.arange(len(particles))

    tf_times = [t[0] * 1000.0 for t in times]
    torch_times_compiled = [t[1] * 1000.0 for t in times]
    jax_times = [t[2] * 1000.0 for t in times]
    tf_times_loop = [t[3] * 1000.0 for t in times]
    
    ax.bar(index, tf_times, bar_width, label='TensorFrost vectorized', alpha=0.8)
    ax.bar(index + bar_width, tf_times_loop, bar_width, label='TensorFrost explicit loop', alpha=0.8)
    ax.bar(index + 2*bar_width, torch_times_compiled, bar_width, label='PyTorch Compiled', alpha=0.8)
    ax.bar(index + 3*bar_width, jax_times, bar_width, label='JAX', alpha=0.8)
    
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Execution Time (ms per iteration)')
    ax.set_title('Performance Comparison on N-body: TensorFrost vs PyTorch vs JAX')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(particles)

    ax.legend()
    plt.grid()
    plt.tight_layout()

    #save plot to file
    plt.savefig('n-body-performance_compile.png')

plot_performance_comparison(times, N0, N1, runs)




    
