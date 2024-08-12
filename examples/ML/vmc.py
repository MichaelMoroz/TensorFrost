import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

tf.initialize(tf.opengl)

lr = 0.005
n_walkers = 2048
opt_steps = 1000
metropolis_per_step = 8
target_acceptance_rate = 0.33
outlier_fraction = 0.4

def Sort(keys, values, element_count):
    tf.region_begin('Sort')
    log2N = tf.ceil(tf.log2(tf.float(element_count)))
    Nround = tf.int(tf.exp2(log2N))
    sort_id = tf.indices([Nround/2])[0]
    steps = tf.int(log2N*(log2N + 1.0)/2.0)

    with tf.loop(steps) as step:
        def getBitonicElementPair(id, step):
            j = tf.floor(tf.sqrt(tf.float(2*step) + 1.0) - 0.5)
            n = tf.round(tf.float(step) - 0.5*j*(j+1.0))
            B = tf.int(tf.round(tf.exp2(j-n)))
            mask = tf.select(n < 0.5, 2*B - 1, B)
            e1 = id%B + 2*B*(id/B)
            e2 = e1 ^ mask
            return e1, e2
        e1, e2 = getBitonicElementPair(sort_id, step)

        with tf.if_cond((e1 < element_count) & (e2 < element_count)):
            key1, key2 = keys[e1], keys[e2]

            #sort by descending order
            with tf.if_cond(key1 < key2):
                val1, val2 = values[e1], values[e2]
                keys[e1] = key2
                keys[e2] = key1
                values[e1] = val2
                values[e2] = val1

    tf.region_end('Sort')
    return keys, values

def sqr(x):
    return x * x

class lognum():
    def __init__(self, value = 0.0, sign = 1.0):
        self.value = value
        self.sign = sign
    
    def asfloat(self):
        return self.sign * tf.exp2(self.value)
    
    def __neg__(self):
        return lognum(self.value, -self.sign)
    
    def __add__(self, other):
        maxv, minv = tf.max(self.value, other.value), tf.min(self.value, other.value)
        diff = maxv - minv
        value = maxv + tf.select(diff > 24.0, 0.0, tf.log2(1.0 + self.sign * other.sign * tf.exp2(-diff)))
        sign = tf.select(self.value > other.value, self.sign, other.sign)
        return lognum(value, sign)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        return lognum(self.value + other.value, self.sign * other.sign)
    
    def __div__(self, other):
        return lognum(self.value - other.value, self.sign * other.sign)
    
    def __pow__(self, other):
        return lognum(self.value * other, self.sign)
    
def aslog(x):
    return lognum(tf.log2(tf.abs(x)), tf.sign(x))

class PSI(tf.Module):
    def __init__(self, atom_n = 1, electron_n = 2):
        super().__init__()
        self.atom_n = atom_n
        self.electron_n = electron_n
        self.params = tf.Parameter([2], tf.float32, requires_grad = False)
        self.step = tf.Parameter([1], tf.int32, requires_grad = False)
        self.seed = tf.Parameter([1], tf.uint32, requires_grad = False)
        self.atoms = tf.Parameter([self.atom_n, 4], tf.float32, requires_grad = False)
        self.weights = tf.Parameter([3], tf.float32)
        self.dx = 5e-4
        self.eps = 1e-6

    def metropolis_dx(self):
        return self.params[0]
    
    def outlier_fraction(self):
        return self.params[1]

    def assert_parameters(self):
        self.seed = tf.assert_tensor(self.seed, [1], tf.uint32)

    def chandrasekhar_helium_psi(self, electrons):
        b, i = tf.indices([electrons.shape[0], 3])
        r12 = tf.norm(electrons[b, 0, i] - electrons[b, 1, i])
        r1 = tf.norm(electrons[b, 0, i] - self.atoms[0, i])
        r2 = tf.norm(electrons[b, 1, i] - self.atoms[0, i])
        alpha = tf.abs(self.weights[0])
        beta = tf.abs(self.weights[1])
        gamma = self.weights[2]
        return (lognum(-alpha*r1-beta*r2) + lognum(-alpha*r2-beta*r1)) * lognum(gamma*r12)

    #computing psi in log space is more numerically stable
    def log_psi(self, electrons):
        psi = self.chandrasekhar_helium_psi(electrons)
        return psi.value

    #get the finite difference gradient and laplacian
    def kinetic_energy(self, e_pos):

        #fd sampling (TODO: use forward mode autodiff)
        n_samples = self.electron_n * 3 * 2 + 1
        b, s, i, c = tf.indices([e_pos.shape[0], n_samples, self.electron_n, 3])
        eid = (s - 1) / 6 #electron index
        did = ((s - 1) / 2) % 3 #dimension index
        sid = (s - 1)  % 2 #forward or backward
        deltaD = tf.select(sid == 0, self.dx, -self.dx)
        pos = tf.unsqueeze(e_pos, axis=1) + tf.select((did == c) & (s > 0) & (eid == i), deltaD, 0.0)
        pos = tf.reshape(pos, [e_pos.shape[0] * n_samples, self.electron_n, 3])

        logpsi = self.log_psi(pos)

        logpsi = tf.reshape(logpsi, [e_pos.shape[0], n_samples])

        #laplacian
        b, = tf.indices([e_pos.shape[0]])
        kinetic = 0.0
        psi_center = logpsi[b, 0]
        for electron in range(self.electron_n):
            for d in range(3):
                psi0 = logpsi[b, 2 * electron * 3 + 2 * d + 1]
                psi1 = logpsi[b, 2 * electron * 3 + 2 * d + 2]
                kinetic += (psi0 + psi1 - 2.0 * psi_center + 0.25*(psi0 - psi1)*(psi0 - psi1)) / (self.dx * self.dx)

        return - 0.5 * kinetic, psi_center
    
    def electron_potential(self, e_pos):
        b, = tf.indices([e_pos.shape[0]])
        V = 0.0

        #compute the nuclei potential sum for each electron*nuclei
        for electron in range(self.electron_n):
            for n in range(self.atom_n):
                r = tf.sqrt(sqr(e_pos[b, electron, 0] - self.atoms[n, 0]) + sqr(e_pos[b, electron, 1] - self.atoms[n, 1]) + sqr(e_pos[b, electron, 2] - self.atoms[n, 2]))
                V -= self.atoms[n, 3] / tf.max(r, self.eps)

        #compute the electron-electron potential sum
        for electron in range(self.electron_n):
            for f in range(electron + 1, self.electron_n):
                r = tf.sqrt(sqr(e_pos[b, electron, 0] - e_pos[b, f, 0]) + sqr(e_pos[b, electron, 1] - e_pos[b, f, 1]) + sqr(e_pos[b, electron, 2] - e_pos[b, f, 2]))
                V += 1.0 / tf.max(r, self.eps)

        return V

    def nuclei_potential(self):
        V = 0.0

        #compute the potential between nuclei
        for n in range(self.atom_n):
            for m in range(n + 1, self.atom_n):
                r = tf.sqrt(sqr(self.atoms[n, 0] - self.atoms[m, 0]) + sqr(self.atoms[n, 1] - self.atoms[m, 1]) + sqr(self.atoms[n, 2] - self.atoms[m, 2]))
                V += self.atoms[n, 3] * self.atoms[m, 3] / tf.max(r, self.eps)

        return V

    def local_energy(self, e_pos):
        kinetic, logpsi = self.kinetic_energy(e_pos)
        return kinetic + self.electron_potential(e_pos) + self.nuclei_potential(), logpsi

    def forward(self, e_pos):
        return self.local_energy(e_pos)

    def energy(self, e_pos):
        energy = self.forward(e_pos)
        mean_energy = tf.mean(energy)
        mean_variance = tf.mean(sqr(energy - mean_energy))
        return mean_energy, mean_variance

    def loss(self, e_pos, _):
        local_energy, logpsi = self.local_energy(e_pos)
        median_ids = self.median_part(local_energy, self.outlier_fraction())
        x_median, psi_median = local_energy[median_ids], logpsi[median_ids]
        return tf.mean(2.0 * (x_median - tf.mean(x_median)).detach_grad() * psi_median)

    def prob_density(self, e_pos):
        return tf.exp(2.0 * self.log_psi(e_pos))

    def inc_step(self):
        self.step += 1

    def rand(self, shape):
        self.seed = tf.pcg(self.seed)

        indices = tf.indices(shape)
        element_index = 0
        for i in range(len(shape)):
            element_index = element_index * shape[i] + indices[i]
        return tf.pcgf(tf.uint(element_index) + self.seed)

    def randn(self, shape):
        x, y = self.rand(shape), self.rand(shape)
        return tf.sqrt(-2.0 * tf.log(x)) * tf.cos(2.0 * np.pi * y)
    
    def median_part(self, x, ratio):
        x_sorted, x_sort_ids = Sort(tf.copy(x), x.indices[0], x.shape[0])
        N = x.shape[0]
        Npart = tf.int(tf.float(N) * ratio * 0.5)   
        i, = tf.indices([N - 2 * Npart])
        return x_sort_ids[Npart + i]

    def metropolis_step(self, e_pos):
        old_prob = self.prob_density(e_pos)

        e_pos_new = e_pos + self.randn(e_pos.shape) * self.metropolis_dx()
        new_prob = self.prob_density(e_pos_new)

        ratio = new_prob / old_prob

        accept = self.rand(ratio.shape) < ratio
        acceptance_rate = tf.mean(tf.float(accept))
        accept = tf.unsqueeze(tf.unsqueeze(accept))
        e_pos = tf.select(accept, e_pos_new, e_pos)

        self.inc_step()
        return e_pos, acceptance_rate

def MetropolisStep():
    wavefunction = PSI()
    wavefunction.initialize_input()

    walkers = tf.input([n_walkers, wavefunction.electron_n, 3], tf.float32)

    new_walkers, acceptance_rate = wavefunction.metropolis_step(walkers)

    params = wavefunction.parameters()
    params.append(new_walkers)
    params.append(acceptance_rate)
    return params

metropolis_step = tf.compile(MetropolisStep)

def ComputeEnergy():
    wavefunction = PSI()
    wavefunction.initialize_input()

    walkers = tf.input([n_walkers, wavefunction.electron_n, 3], tf.float32)

    return wavefunction.forward(walkers)

compute_energy = tf.compile(ComputeEnergy)

def GetModelOptimizer():
    wavefunction = PSI()
    optimizer = tf.optimizers.adam(wavefunction, lr, beta1 = 0.0)
    return optimizer, wavefunction

def OptimizeEnergy():
    optimizer, wavefunction = GetModelOptimizer()
    optimizer.initialize_input()

    walkers = tf.input([n_walkers, wavefunction.electron_n, 3], tf.float32)
    loss = optimizer.step(walkers, None)
    params = optimizer.parameters()
    params.append(loss)
    return params

optimize_energy = tf.compile(OptimizeEnergy)

optimizer, wavefunction = GetModelOptimizer()
optimizer.initialize_parameters()
optimizer.net.params = tf.tensor(np.array([0.5, outlier_fraction]).astype(np.float32))
optimizer.net.atoms = tf.tensor(np.array([[0.0, 0.0, 0.0, 2.0]]).astype(np.float32))
optimizer.net.step = tf.tensor(np.array([0], np.int32))
optimizer.net.seed = tf.tensor(np.array([0], np.uint32))

#np.random.seed(0)
walkers = np.random.randn(n_walkers, wavefunction.electron_n, 3).astype(np.float32)
walkers_tf = tf.tensor(walkers)

acceptance_rate_history = []
energy_history = []

def list_parameters(optimizer):
    print("List of parameters:")
    for i, param in enumerate(optimizer.parameters()):
        print("Parameter ", i, ": ", param.numpy)

progress_bar = tqdm(range(opt_steps))
smoothed_loss = 0.0
smoothed_energy = 0.0
smoothed_acceptance_rate = 0.0
smoothing = 0.9
for i in progress_bar:
    if(i == 0): tf.renderdoc_start_capture()

    for j in range(metropolis_per_step):
        out = metropolis_step(wavefunction, walkers_tf)
        walkers_tf = out[-2]
        acceptance_rate = out[-1]
        wavefunction.update_parameters(out[:-2])
        
        acceptance_rate = acceptance_rate.numpy
        smoothed_acceptance_rate = smoothing * smoothed_acceptance_rate + (1.0 - smoothing) * acceptance_rate[0]
        acceptance_rate_history.append(smoothed_acceptance_rate)
        cur_params = wavefunction.params.numpy
        if(acceptance_rate[0] < target_acceptance_rate):
            cur_params[0] *= 0.98
        if(acceptance_rate[0] > target_acceptance_rate):
            cur_params[0] *= 1.02

        if(i > 300): cur_params[1] = 0.1

        wavefunction.params = tf.tensor(cur_params)

    out = optimize_energy(optimizer, walkers_tf)
    optimizer.update_parameters(out[:-1])
    loss = out[-1].numpy

    local_energy, logpsi = compute_energy(wavefunction, walkers_tf)
    local_energy = local_energy.numpy
    mean_energy = np.mean(local_energy)
    smoothed_energy = smoothing * smoothed_energy + (1.0 - smoothing) * mean_energy
    smoothed_loss = smoothing * smoothed_loss + (1.0 - smoothing) * loss
    energy_history.append(smoothed_energy)

    progress_bar.set_postfix(acceptance_rate = smoothed_acceptance_rate, loss = smoothed_loss, energy = mean_energy)

    if(i == 0): tf.renderdoc_end_capture()

#compute the final energy
local_energy, logpsi = compute_energy(wavefunction, walkers_tf)
local_energy = local_energy.numpy
walkers = walkers_tf.numpy
sorted_args = np.argsort(local_energy)
sorted_walkers = walkers[sorted_args]
sorted_energy = local_energy[sorted_args]

print("Final Energy: ", np.mean(smoothed_energy))
#print weights
print("Weights: ", wavefunction.weights.numpy)

# eN = sorted_energy.shape[0]
# e0 = sorted_energy[eN // 2 - 32]
# e1 = sorted_energy[eN // 2 + 32]
# dE = e1 - e0
# di = 32 * 2
# #median line plot: E = (e0 + e1) / 2 + (i - N/2) * dE / di


# #plot sorted energy
# plt.figure()
# plt.plot(sorted_energy)
# plt.plot(np.arange(eN), (e0 + e1) * 0.5 + (np.arange(eN) - eN * 0.5) * dE / di, 'r')
# plt.xlabel('Walker')
# plt.ylabel('Energy, Hartree')
# plt.title('Sorted Energy')
# plt.grid(True)
# plt.show()

# Create a figure with more vertical space
fig = plt.figure(figsize=(10, 10))

# Create subplot grid with more vertical space between plots
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4)

# Plot 1: Energy history (1x2 slot)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(energy_history)
ax1.set_xlabel('Step')
ax1.set_ylabel('Energy, Hartree')
ax1.grid(True)
ax1.set_title('Energy History', pad=20)

# Plot 2: Energy histogram (1x2 slot)
ax2 = fig.add_subplot(gs[1, :])
ax2.hist(local_energy, bins=1000)
ax2.set_xlabel('Energy, Hartree')
ax2.set_ylabel('Frequency')
ax2.set_yscale('log')
ax2.grid(True)
ax2.set_title('Energy Histogram', pad=20)

n_outliers = int(n_walkers * outlier_fraction * 0.5)
outliers_bottom = sorted_walkers[:n_outliers]
outliers_top = sorted_walkers[-n_outliers:]

# Plot 3: Walkers in x-y plane (1x1 slot)
ax3 = fig.add_subplot(gs[2, 0])
all_walkers = walkers.reshape(-1, 3)
ax3.scatter(all_walkers[:, 0], all_walkers[:, 1], s=1)
outliers = np.concatenate((outliers_bottom, outliers_top))
outliers = outliers.reshape(-1, 3)
ax3.scatter(outliers[:, 0], outliers[:, 1], s=1.5, c='red')
ax3.set_xlabel('x, a.u.')
ax3.set_ylabel('y, a.u.')
ax3.grid(True)
ax3.set_title('Walkers in x-y Plane', pad=20)

# Plot 4: Difference of electron positions (1x1 slot)
ax4 = fig.add_subplot(gs[2, 1])
diff = walkers[:, 0] - walkers[:, 1]
diff = diff.reshape(-1, 3)
ax4.scatter(diff[:, 0], diff[:, 1], s=1)
diff_outlier = np.concatenate((outliers_bottom[:, 0] - outliers_bottom[:, 1], outliers_top[:, 0] - outliers_top[:, 1]))
diff_outlier = diff_outlier.reshape(-1, 3)
ax4.scatter(diff_outlier[:, 0], diff_outlier[:, 1], s=1.5, c='red')
ax4.set_xlabel('x, a.u.')
ax4.set_ylabel('y, a.u.')
ax4.grid(True)
ax4.set_title('Difference of Electron Positions', pad=20)

# Adjust layout
plt.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.92, hspace=0.4)

plt.show()