import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt

tf.initialize(tf.cpu)

class PSI(tf.Module):
    def __init__(self, atom_n = 1, electron_n = 2):
        super().__init__()
        self.atom_n = atom_n
        self.electron_n = electron_n
        self.params = tf.Parameter([1], tf.float32, requires_grad = False)
        self.step = tf.Parameter([1], tf.int32, requires_grad = False)
        self.seed = tf.Parameter([1], tf.uint32, requires_grad = False)
        self.atoms = tf.Parameter([self.atom_n, 4], tf.float32, requires_grad = False)
        self.weights = tf.Parameter([3], tf.float32)
        self.dx = 1e-2
        self.eps = 1e-6

    def metropolis_dx(self):
        return self.params[0]

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
        return (tf.exp(-alpha*r1-beta*r2)+tf.exp(-alpha*r2-beta*r1))*(1.0 + gamma*r12)
    
    #computing psi in log space is more numerically stable
    def log_psi(self, electrons):
        psi = self.chandrasekhar_helium_psi(electrons)
        return tf.log(tf.max(tf.abs(psi),self.eps))
    
    #get the finite difference gradient and laplacian
    def kinetic_energy(self, e_pos):
        
        #fd sampling (TODO: use forward mode autodiff)
        # n_samples = self.electron_n * 3 * 2 + 1
        # b, s, i, c = tf.indices([e_pos.shape[0], n_samples, self.electron_n, 3])
        # eid = (s - 1) / 6 #electron index
        # did = ((s - 1) / 2) % 3 #dimension index
        # sid = (s - 1)  % 2 #forward or backward
        # deltaD = tf.select(sid == 0, self.dx, -self.dx)
        # pos = tf.unsqueeze(e_pos, axis=1) + tf.select((did == c) & (s > 0) & (eid == i), deltaD, 0.0)
        # pos = tf.reshape(pos, [e_pos.shape[0] * n_samples, self.electron_n, 3])

        # logpsi = self.log_psi(pos)

        # logpsi = tf.reshape(logpsi, [e_pos.shape[0], n_samples])

        # #laplacian
        # b, = tf.indices([e_pos.shape[0]])
        # kinetic = 0.0
        # psi_center = logpsi[b, 0]
        # for electron in range(self.electron_n):
        #     for d in range(3):
        #         psi0 = logpsi[b, 2 * electron * 3 + 2 * d + 1]
        #         psi1 = logpsi[b, 2 * electron * 3 + 2 * d + 2]
        #         kinetic += - 0.5 * (psi0 + psi1 - 2.0 * psi_center + 0.25*(psi0 - psi1)*(psi0 - psi1)) / (self.dx * self.dx)

        psi_center = self.log_psi(e_pos)
        b, i, j = e_pos.indices
        kinetic = 0.0
        for electron in range(self.electron_n):
            for d in range(3):
                psi0 = self.log_psi(e_pos + tf.select((i == electron) & (j == d), self.dx, 0.0))
                psi1 = self.log_psi(e_pos - tf.select((i == electron) & (j == d), self.dx, 0.0))
                kinetic += - 0.5 * (psi0 + psi1 - 2.0 * psi_center + 0.25*(psi0 - psi1)*(psi0 - psi1)) / (self.dx * self.dx)

        return kinetic

    def electron_potential(self, e_pos):
        b, = tf.indices([e_pos.shape[0]])
        V = 0.0

        #compute the nuclei potential sum for each electron*nuclei
        for electron in range(self.electron_n):
            for n in range(self.atom_n):
                r = tf.sqrt((e_pos[b, electron, 0] - self.atoms[n, 0])**2.0 + (e_pos[b, electron, 1] - self.atoms[n, 1])**2.0 + (e_pos[b, electron, 2] - self.atoms[n, 2])**2.0)
                V -= self.atoms[n, 3] / tf.max(r, self.eps)

        #compute the electron-electron potential sum
        for electron in range(self.electron_n):
            for f in range(electron + 1, self.electron_n):
                r = tf.sqrt((e_pos[b, electron, 0] - e_pos[b, f, 0])**2.0 + (e_pos[b, electron, 1] - e_pos[b, f, 1])**2.0 + (e_pos[b, electron, 2] - e_pos[b, f, 2])**2.0)
                V += 1.0 / tf.max(r, self.eps)

        return V
    
    def nuclei_potential(self):
        V = 0.0

        #compute the potential between nuclei
        for n in range(self.atom_n):
            for m in range(n + 1, self.atom_n):
                r = tf.sqrt((self.atoms[n, 0] - self.atoms[m, 0])**2 + (self.atoms[n, 1] - self.atoms[m, 1])**2 + (self.atoms[n, 2] - self.atoms[m, 2])**2)
                V += self.atoms[n, 3] * self.atoms[m, 3] / tf.max(r, self.eps)

        return V
    
    def local_energy(self, e_pos):
        return self.kinetic_energy(e_pos) + self.electron_potential(e_pos) + self.nuclei_potential()

    def forward(self, e_pos):
        return self.local_energy(e_pos)
    
    def energy(self, e_pos):
        return self.batched_mean(self.forward(e_pos))

    def loss(self, e_pos, _):
        local_energy = self.forward(e_pos)
        return self.variance(local_energy)

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
    
    def split_dim(self, x, M):
        N = x.shape[0]
        i, j = tf.indices([N/M, M])
        return x[i * M + j]

    def batched_mean(self, x):
        batch_size = 128
        x_reshaped = tf.reshape(x, [x.shape[0] / batch_size, batch_size]) #self.split_dim(x, batch_size) 
        return tf.mean(tf.mean(x_reshaped))
    
    def variance(self, x):
        x_mean = tf.mean(x)
        return tf.mean((x - x_mean) * (x - x_mean)) + x_mean

    def metropolis_step(self, e_pos):
        old_prob = self.prob_density(e_pos)

        e_pos_new = e_pos + self.randn(e_pos.shape) * self.metropolis_dx()
        new_prob = self.prob_density(e_pos_new)

        ratio = new_prob / old_prob

        accept = self.rand(ratio.shape) < ratio
        acceptance_rate = self.batched_mean(tf.float(accept))
        accept = tf.unsqueeze(tf.unsqueeze(accept))
        e_pos = tf.select(accept, e_pos_new, e_pos)

        self.inc_step()
        return e_pos, acceptance_rate

def MetropolisStep():
    wavefunction = PSI()
    wavefunction.initialize_input()

    walkers = tf.input([-1, wavefunction.electron_n, 3], tf.float32)
    n_walkers = walkers.shape[0]

    new_walkers, acceptance_rate = wavefunction.metropolis_step(walkers)

    params = wavefunction.parameters()
    params.append(new_walkers)
    params.append(acceptance_rate)
    return params

metropolis_step = tf.compile(MetropolisStep)

def ComputeEnergy():
    wavefunction = PSI()
    wavefunction.initialize_input()

    walkers = tf.input([-1, wavefunction.electron_n, 3], tf.float32)
    n_walkers = walkers.shape[0]

    return wavefunction.energy(walkers)

metropolis_step = tf.compile(MetropolisStep)

lr = 0.001

def OptimizeEnergy():
    wavefunction = PSI()
    optimizer = tf.optimizers.sgd(wavefunction, lr)
    optimizer.initialize_input()

    walkers = tf.input([-1, wavefunction.electron_n, 3], tf.float32)
    n_walkers = walkers.shape[0]

    loss = optimizer.step(walkers, None)
    params = optimizer.parameters()
    params.append(loss)
    return params 

optimize_energy = tf.compile(OptimizeEnergy)

wavefunction = PSI()
optimizer = tf.optimizers.sgd(wavefunction, lr)
optimizer.initialize_parameters()
optimizer.net.params = tf.tensor(np.array([0.5]).astype(np.float32))
optimizer.net.atoms = tf.tensor(np.array([[0.0, 0.0, 0.0, 2.0]]).astype(np.float32))
optimizer.net.weights = tf.tensor(np.array([1.0, 1.2, 0.3]))  #tf.tensor(np.array([1.5185, 2.2154, 0.3604]))
optimizer.net.step = tf.tensor(np.array([0], np.int32))
optimizer.net.seed = tf.tensor(np.array([0], np.uint32))

n_walkers = 16384
n_steps = 2048
n_steps_per_optimization = 16
target_acceptance_rate = 0.4

params = wavefunction.parameters()
for param in params:
    print(param.numpy)

walkers = 0.25*np.random.randn(n_walkers, wavefunction.electron_n, 3).astype(np.float32)
walkers_tf = tf.tensor(walkers)

acceptance_rate_history = []
energy_history = []

def list_parameters(optimizer):
    print("List of parameters:")
    for i, param in enumerate(optimizer.parameters()):
        print("Parameter ", i, ": ", param.numpy)
for i in range(n_steps):
    if(i == 0): tf.renderdoc_start_capture()

    out = metropolis_step(optimizer.net, walkers_tf)
    walkers_tf = out[-2]
    acceptance_rate = out[-1]
    optimizer.net.update_parameters(out[:-2])

    acceptance_rate = acceptance_rate.numpy
    acceptance_rate_history.append(acceptance_rate[0])
    cur_params = wavefunction.params.numpy
    if(acceptance_rate[0] < target_acceptance_rate):
        cur_params[0] *= 0.98
    if(acceptance_rate[0] > target_acceptance_rate):
        cur_params[0] *= 1.02
    wavefunction.params = tf.tensor(cur_params)

    if(i % n_steps_per_optimization == 0):
        out = optimize_energy(optimizer, walkers_tf)
        optimizer.update_parameters(out[:-1])
        loss = out[-1].numpy
        energy_history.append(loss)
        print("Step: ", i, " Energy: ", loss)

    if(i == 1): tf.renderdoc_end_capture()

list_parameters(optimizer)

#print average acceptance rate
print("Average Acceptance Rate: ", np.mean(acceptance_rate_history))

#plot acceptance rate
plt.plot(acceptance_rate_history)
plt.xlabel('Step')
plt.ylabel('Acceptance Rate')
plt.grid()
plt.show()

#plot energy history
plt.plot(energy_history)
plt.xlabel('Step')
plt.ylabel('Energy')
plt.grid()
plt.show()



