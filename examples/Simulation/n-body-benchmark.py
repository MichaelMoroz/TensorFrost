import numpy as np
import TensorFrost as tf
import torch
import imageio
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

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

nbody_tf = tf.compile(n_body)

#if hasattr(torch, 'compile'):
#    n_body_torch = torch.compile(n_body_torch)

def performance_comparison(iters=1000, N=1000):
    X_tf = 5.0 * np.random.randn(N, 3).astype(np.float32)
    V_tf = np.zeros((N, 3), dtype=np.float32)

    # PyTorch setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_torch = 5.0 * torch.randn(N, 3, device=device, dtype=torch.float32)
    V_torch = torch.zeros((N, 3), device=device, dtype=torch.float32)

    # TensorFrost timing
    start_time = time.time()
    for _ in range(iters):
        X_tf, V_tf = nbody_tf(X_tf, V_tf)
    X_np = X_tf.numpy 
    tf_time = time.time() - start_time

    # PyTorch timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iters):
            X_torch, V_torch = n_body_torch(X_torch, V_torch)
    torch_time = time.time() - start_time

    return torch_time / iters, tf_time / iters

#performance_comparison(1000)

# Do performance comparisons for different numbers of particles
N0 = 100
N1 = 10000
runs = 6

torch_times = []
tf_times = []

for N in tqdm(range(N0, N1, N1//runs)):
    speedup = performance_comparison(1000000//N, N)
    torch_times.append(speedup[0])
    tf_times.append(speedup[1])

def plot_performance_comparison(torch_times, tf_times, N0, N1, runs):
    particles = np.linspace(N0, N1, runs)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.35
    index = np.arange(len(particles))
    
    ax.bar(index, tf_times, bar_width, label='TensorFrost', alpha=0.8)
    ax.bar(index + bar_width, torch_times, bar_width, label='PyTorch', alpha=0.8)
    
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Performance Comparison: TensorFrost vs PyTorch')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([f'{int(p)}' for p in particles], rotation=45)
    
    ax.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

plot_performance_comparison(torch_times, tf_times, N0, N1, runs)




    
