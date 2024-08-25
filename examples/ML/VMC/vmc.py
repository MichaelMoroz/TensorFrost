import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from tqdm import tqdm
import time

from utils import *
from atom import *
from logdet import *
from molecules import *
from visualizer import *

tf.initialize(tf.opengl)

register_logdet()

lr0 = 0.005
lr1 = 0.001
reg0 = 0.01
reg1 = 0.005
clip0 = 0.3
clip1 = 0.1
n_walkers = 2048
opt_steps = 8000

pause_training = False
custom_hyperparameters = False
custom_lr = lr0
custom_reg = reg0
custom_clip = clip0

def scheduler(step):
    t = min(max(float(step) / float(opt_steps), 0.0), 1.0)
    cur_lr = lr0 * (1.0 - t) + lr1 * t
    cur_reg = reg0 * (1.0 - t) + reg1 * t
    cur_clip = clip0 * (1.0 - t) + clip1 * t
    if custom_hyperparameters:
        cur_lr = custom_lr
        cur_reg = custom_reg
        cur_clip = custom_clip
    if pause_training:
        cur_lr = 0.0
    return cur_lr, cur_reg, cur_clip

#how many metropolis steps to perform per optimization step
metropolis_per_step = 12

#how many metropolis steps to store in the history to render
metropolis_history_length = 128

#what target fraction of the walkers to move in each step
target_acceptance_rate = 0.4

#what fraction of the walkers [sorted by local energy] to ignore in the loss function
#outliers on the tails of the distribution can cause the optimization to diverge due to numerical instability
#but I found that sorting and ignoring a fixed fraction of the sorted walkers is simpler and seems to work well
outlier_fraction = 0.15
#range for clipping around the median of the local energy from ferminet/psiformer (in variance units)
mad_range = 3.0

smoothing = 0.995

molecule = ch4_molecule

molecule_info = molecule.get_summary()
print(molecule_info)

target_energy = molecule.target_energy
atoms = molecule.get_atoms()
orbitals = molecule.get_orbitals(2, 2)
print(orbitals)
atom_n = atoms.shape[0]
electron_n = molecule.electron_count
spin_up_n = molecule.spin_up_electrons

class PSI(tf.Module):
    def __init__(self):
        super().__init__()
        self.atom_n = atom_n
        self.electron_n = electron_n
        self.spin_up_n = spin_up_n
        self.spin_down_n = electron_n - spin_up_n
        self.determinants = 1
        self.orbital_n = orbitals.shape[0]
        self.mid_n = 16
        self.corr_n = 12

        # Parameters
        self.params = tf.Parameter([2], tf.float32, requires_grad = False)
        self.step = tf.Parameter([1], tf.int32, requires_grad = False)
        self.seed = tf.Parameter([1], tf.uint32, requires_grad = False)
        self.atoms = tf.Parameter([self.atom_n, 4], tf.float32, requires_grad = False)
        self.orbital_atom_id = tf.Parameter([self.orbital_n], tf.int32, requires_grad = False)
        self.probability = tf.Parameter([n_walkers], tf.float32, requires_grad = False)

        # Weights
        self.orbi_layer0 = tf.Parameter([self.orbital_n, 4], tf.float32, random_scale = 0.025)
        self.orbi_layer0_bias = tf.Parameter([self.orbital_n], tf.float32, random_scale = 1.0)
        self.envelope_layer = tf.Parameter([self.orbital_n], tf.float32, random_scale = 3.0)

        self.orbi_layer1 = tf.Parameter([self.orbital_n, self.mid_n], tf.float32)
        self.mid_layer1 = tf.Parameter([self.mid_n, self.mid_n], tf.float32)
        self.mid_layer1_bias = tf.Parameter([self.mid_n], tf.float32, random_scale = 1.0)
        self.mid_layer2 = tf.Parameter([self.mid_n, self.mid_n], tf.float32)
        self.mid_layer3 = tf.Parameter([self.mid_n, self.mid_n], tf.float32)
        self.mid_layer3_bias = tf.Parameter([self.mid_n], tf.float32, random_scale = 1.0)

        self.exchange_layer_out_up = tf.Parameter([self.mid_n, self.corr_n], tf.float32)
        self.exchange_layer_out_down = tf.Parameter([self.mid_n, self.corr_n], tf.float32)
        self.exchange_layer_in = tf.Parameter([self.corr_n*2, self.mid_n], tf.float32)

        self.up_layer = tf.Parameter([self.mid_n, self.determinants * self.spin_up_n], tf.float32)
        self.down_layer = tf.Parameter([self.mid_n, self.determinants * self.spin_down_n], tf.float32)
        self.gamma = tf.Parameter([4], tf.float32)
        self.dx = 5e-3
        self.eps = 1e-7

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

    def eye(self, N):
        i, j = tf.indices([N, N])
        return tf.select(i == j, 1.0, 0.0)

    def eye_like(self, X):
        i, j = X.indices
        return tf.select(i == j, 1.0, 0.0)

    def get_spin_up(self, data):
        upi = tf.indices([data.shape[0], self.spin_up_n, data.shape[2]])
        return data[upi[0], upi[1], upi[2]]
    
    def get_spin_down(self, data):
        downi = tf.indices([data.shape[0], self.spin_down_n, data.shape[2]])
        return data[downi[0], self.spin_up_n + downi[1], downi[2]]
    
    def concat_exchange(self, exchange_up, exchange_down):
        idx = tf.indices([exchange_up.shape[0], exchange_up.shape[1], self.corr_n*2])
        return tf.select(idx[2] < exchange_up.shape[2], exchange_up[idx[0], idx[1], idx[2]], exchange_down[idx[0], idx[1], idx[2] - exchange_up.shape[2]])

    def psi_new(self, electrons):
        tf.region_begin('Psi')

        tf.region_begin('AtomOrbitals')
        #compute electron-atom relative positions and distances [batch, electron, orbital_features, 3] + [batch, electron, orbital_features]
        b, e, a, d = tf.indices([electrons.shape[0], self.electron_n, self.orbital_n, 3])
        atom_id = self.orbital_atom_id[a]
        ri = tf.unsqueeze(electrons, axis=-2) - self.atoms[atom_id, d]
        r = tf.norm(ri)

        #concatenate ri and r [batch, electron, orbital_features, 3 + 1]
        b, e, a, d = tf.indices([electrons.shape[0], self.electron_n, self.orbital_n,  4])
        in0 = tf.select(d < 3, ri[b, e, a, d], r[b, e, a]) 

        #orbitals around atoms [batch, electron, orbital_features]
        out0 = (tf.dot(in0, self.orbi_layer0) + self.orbi_layer0_bias)
        atom_orbitals = out0 * tf.exp(-r * tf.abs(self.envelope_layer))
        tf.region_end('AtomOrbitals')

        tf.region_begin('Orbitals')
        atom_orbitals = tf.reshape(atom_orbitals, atom_orbitals.shape) #force compiler to not fuse with previous operations
        orbitals = atom_orbitals @ self.orbi_layer1 
        orbitals = orbitals*tf.tanh(self.mid_layer1_bias + orbitals @ self.mid_layer1)

        #exchange layer for electron correlation
        exchange_out_up = self.get_spin_up(orbitals) @ self.exchange_layer_out_up
        exchange_out_down = self.get_spin_down(orbitals) @ self.exchange_layer_out_down
        exchange_in_up = tf.unsqueeze(tf.mean(exchange_out_up, axis=1), axis=1)
        exchange_in_down = tf.unsqueeze(tf.mean(exchange_out_down, axis=1), axis=1)
        exchange_in = self.concat_exchange(exchange_in_up, exchange_in_down)
        orbitals = orbitals + (orbitals @ self.mid_layer2) * tf.tanh(exchange_in @ self.exchange_layer_in)

        orbitals = orbitals*tf.tanh(self.mid_layer3_bias + orbitals @ self.mid_layer3)

        #up and down features
        #note: since we didn't use biases and multiplied the correlation layer with the orbitals, we dont really need an envelope function
        #the orbitals should already satisfy the boundary conditions
        up_features = self.get_spin_up(orbitals) @ self.up_layer
        down_features = self.get_spin_down(orbitals) @ self.down_layer
        tf.region_end('Orbitals')

        tf.region_begin('SlaterDeterminant')

        #compute determinant for each set of electron orbitals [batch, determinant]
        #todo: output sign of the determinants
        det_down = self.logdeterminant(down_features, self.spin_down_n)
        det_up = self.logdeterminant(up_features, self.spin_up_n)

        #product of spin up and spin down determinants
        orbiprod = det_up + det_down
        
        #orbiprod = det_up * det_down
        #orbiprod = tf.squeeze(det_up) + tf.squeeze(det_down)
        #orbiprod = tf.squeeze(det_up + det_down)
        #orbiprodlog = orbiprod.value
        #orbiprodsign = orbiprod.sign
        #sum the determinants (use logsumexp trick)  
        #maxorbiprodlog = tf.sum(orbiprodlog)
        #detsum = tf.sum(tf.exp2(orbiprodlog - tf.unsqueeze(maxorbiprodlog)))
        #orbisum = maxorbiprodlog + tf.log2(tf.abs(detsum))

        tf.region_end('SlaterDeterminant')

        tf.region_begin('Jastrow')
        psi = lognum(orbiprod) * self.jastrow(electrons)
        tf.region_end('Jastrow')

        tf.region_end('Psi')
        return psi
    
    def jastrow(self, electrons):
        #compute the jastrow factor for each pair of electrons
        electron_pair_count = self.electron_n * (self.electron_n - 1) // 2
        b, e, c = tf.indices([electrons.shape[0], electron_pair_count, 3])
        #row and column of the triangular matrix
        i = tf.int((tf.sqrt(8.0 * tf.float(e) + 1.0) + 1.0) / 2.0)
        j = e - i * (i - 1) / 2
        rij = tf.norm(electrons[b, i, c] - electrons[b, j, c])
        
        b, e = tf.indices([electrons.shape[0], electron_pair_count])
        i = tf.int((tf.sqrt(8.0 * tf.float(e) + 1.0) + 1.0) / 2.0)
        j = e - i * (i - 1) / 2
        is_up_i = i < self.spin_up_n
        is_up_j = j < self.spin_up_n
        param1 = tf.select(is_up_i == is_up_j, self.gamma[0], self.gamma[1])
        param2 = tf.select(is_up_i == is_up_j, self.gamma[2], self.gamma[3])
        jastrowcomp = - tf.abs(param1) / (1.0 + tf.abs(param2) * rij)
        jastrow = tf.sum(jastrowcomp)

        return lognum(jastrow)

    #unrolled determinant calculation (not exactly the best way to do this, but easy auto-differentiation)
    def logdeterminant(self, orbitals, N):
        orbitals = tf.reshape(orbitals, [orbitals.shape[0], N, N, self.determinants])

        bi, i, j = tf.indices([orbitals.shape[0] * self.determinants, N, N])
        b = bi / self.determinants
        d = bi % self.determinants

        reordered = orbitals[b, i, j, d]
        logdet = tf.custom("logdet", [reordered], [reordered.shape[0]])
        logdet = tf.reshape(logdet, [orbitals.shape[0], self.determinants])
        logdet = tf.squeeze(logdet)
        return logdet / 0.69314718056
        return tf.squeeze(logdet / 0.69314718056)

    #computing psi in log space is more numerically stable
    def log_psi(self, electrons):
        psi = self.psi_new(electrons)
        return psi.log()

    #get the finite difference gradient and laplacian
    def kinetic_energy(self, e_pos):
        logpsi_center = self.log_psi(e_pos)

        tf.region_begin('Laplacian')
        #fd sampling (TODO: use forward mode autodiff)
        n_samples = self.electron_n * 3 * 2
        b, s, i, c = tf.indices([e_pos.shape[0], n_samples, self.electron_n, 3])
        eid = s / 6 #electron index
        did = (s / 2) % 3 #dimension index
        sid = s % 2 #forward or backward
        deltaD = tf.select(sid == 0, self.dx, -self.dx)
        pos = tf.unsqueeze(e_pos, axis=1) + tf.select((did == c) & (eid == i), deltaD, 0.0)
        pos = tf.reshape(pos, [e_pos.shape[0] * n_samples, self.electron_n, 3])
        logpsi = self.log_psi(pos)
        logpsi = tf.reshape(logpsi, [e_pos.shape[0], n_samples])

        #laplacian of the wavefunction divided by the wavefunction (which is the laplacian + the gradient squared of the log wavefunction)
        b, = tf.indices([e_pos.shape[0]])
        kinetic = 0.0
        for electron in range(self.electron_n):
            for d in range(3):
                psi0 = logpsi[b, 6 * electron + 2 * d + 0]
                psi1 = logpsi[b, 6 * electron + 2 * d + 1]
                kinetic += (psi0 + psi1 - 2.0 * logpsi_center + 0.25*(psi0 - psi1)*(psi0 - psi1)) / (self.dx * self.dx)

        tf.region_end('Laplacian')
        return - 0.5 * kinetic, logpsi_center
    
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

    def loss(self, e_pos):
        local_energy, logpsi = self.local_energy(e_pos)
        ids = local_energy.indices[0]
        sample_count = local_energy.shape[0]
        x_sorted, x_sort_ids = Sort(tf.copy(local_energy), ids, sample_count)
        x_sorted, psi_sorted = local_energy[x_sort_ids], logpsi[x_sort_ids]

        #compute the mask for the central fraction of the sorted samples
        fraction = tf.int(self.outlier_fraction() * tf.float(sample_count) * 0.5)
        num_masked = sample_count - 2 * fraction
        median_mask = tf.select((ids >= fraction) & (ids < (sample_count - fraction)), tf.float(sample_count)/tf.float(num_masked), 0.0)

        #clip the local energy to a region around the median
        median = x_sorted[sample_count / 2]
        mad_median = tf.mean(tf.abs(x_sorted - median)*median_mask)
        x_clipped = tf.clamp(x_sorted, median - mad_range * mad_median, median + mad_range * mad_median)

        x_mean = tf.mean(x_clipped*median_mask)
        return tf.mean(2.0 * (x_clipped - x_mean).detach_grad() * psi_sorted * median_mask), x_mean

    def log_prob_density(self, e_pos):
        return 2.0 * self.log_psi(e_pos)

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


    def metropolis_step(self, e_pos, update_prob):
        #do we only update the probability?
        update_prob = update_prob > 0

        #reuse the previous step probability for better performance
        old_prob = self.probability #self.log_prob_density(e_pos)

        e_pos_new = e_pos + tf.select(update_prob, 0.0, self.randn(e_pos.shape) * self.metropolis_dx())
        new_prob = self.log_prob_density(e_pos_new)

        ratio = tf.exp(tf.clamp(new_prob - old_prob, -50.0, 50.0))

        accept = (self.rand(ratio.shape) < ratio)

        #update the probability
        self.probability = tf.select(accept | update_prob, new_prob, old_prob)

        acceptance_rate = tf.mean(tf.float(accept))
        accept = tf.unsqueeze(tf.unsqueeze(accept))

        #update the electron positions
        e_pos = tf.select(accept & (~update_prob), e_pos_new, e_pos)

        self.inc_step()
        return e_pos, acceptance_rate

def MetropolisStep():
    wavefunction = PSI()
    wavefunction.initialize_input()

    walkers = tf.input([n_walkers, wavefunction.electron_n, 3], tf.float32)
    allsteps_walkers = tf.input([-1, wavefunction.electron_n, 3], tf.float32)
    metropolis_params = tf.input([2], tf.int32)
    update_prob = metropolis_params[0]
    cur_step = metropolis_params[1]

    new_walkers, acceptance_rate = wavefunction.metropolis_step(walkers, update_prob)

    i, j, k = new_walkers.indices
    allsteps_walkers[i * metropolis_history_length + cur_step, j, k] = new_walkers

    params = wavefunction.parameters()
    params.append(new_walkers)
    params.append(acceptance_rate)
    return params

metropolis_step = tf.compile(MetropolisStep)

def GetModelOptimizer():
    wavefunction = PSI()
    optimizer = tf.optimizers.adam(wavefunction, beta1 = 0.0, beta2 = 0.999, reg_type = tf.regularizers.l2, reg = 0.02, clip = 0.01)
    #optimizer = tf.optimizers.sgd(wavefunction, reg_type = tf.regularizers.l2, reg = 0.02, clip = 0.01)
    optimizer.set_clipping_type(tf.clipping.norm)
    return optimizer, wavefunction

def OptimizeEnergy():
    optimizer, wavefunction = GetModelOptimizer()
    optimizer.initialize_input()

    walkers = tf.input([n_walkers, wavefunction.electron_n, 3], tf.float32)
    in_params = tf.input([3], tf.float32)
    optimizer.learning_rate = in_params[0]
    optimizer.reg = in_params[1]
    optimizer.clip = in_params[2]
    loss, Emean = wavefunction.loss(walkers)
    optimizer.step(loss)
    params = optimizer.parameters()
    params.append(loss)
    params.append(Emean)
    return params

optimize_energy = tf.compile(OptimizeEnergy)

np.random.seed(1)

optimizer, wavefunction = GetModelOptimizer()
optimizer.initialize_parameters()
optimizer.net.params = tf.tensor(np.array([0.5, outlier_fraction]).astype(np.float32))
optimizer.net.atoms = tf.tensor(atoms)
optimizer.net.orbital_atom_id = tf.tensor(orbitals)
optimizer.net.step = tf.tensor(np.array([0], np.int32))
optimizer.net.seed = tf.tensor(np.array([0], np.uint32))

class Status:
    def __init__(self):
        self.acceptance_rate_history = []
        self.energy_history = []
        self.smoothed_loss = 0.0
        self.smoothed_energy = 0.0
        self.smoothed_acceptance_rate = 0.0
        self.improvement = 0.0

status = Status()

def OptimizationStep(i, optimizer, walkers_tf, walker_history_tf, status):

    start_time = time.time()

    for j in range(metropolis_per_step):
        update_prob = 1 if (j == 0) else 0 #must update the stored probability in the first step, since wavefunction has changed
        total_step = i * metropolis_per_step + j
        history_step = total_step % metropolis_history_length
        metropolis_params = [update_prob, history_step]
        out = metropolis_step(optimizer.net, walkers_tf, walker_history_tf, metropolis_params)
        walkers_tf = out[-2]
        acceptance_rate = out[-1]
        optimizer.net.update_parameters(out[:-2])
        
        acceptance_rate = acceptance_rate.numpy
        status.smoothed_acceptance_rate = smoothing * status.smoothed_acceptance_rate + (1.0 - smoothing) * acceptance_rate[0]
        status.acceptance_rate_history.append(status.smoothed_acceptance_rate)
        cur_params = optimizer.net.params.numpy

        #update the metropolis step size based on the acceptance rate
        if(acceptance_rate[0] < target_acceptance_rate):
            cur_params[0] *= 0.98
        if(acceptance_rate[0] > target_acceptance_rate):
            cur_params[0] *= 1.02

        optimizer.net.params = tf.tensor(cur_params)

    metropolis_step_time = time.time() - start_time
    start_time = time.time()

    cur_lr, cur_reg, cur_clip = scheduler(i)

    loss, Emean = 0.0, 0.0
    if cur_lr > 0.0:
        if(status.improvement < 0.0):
            cur_lr *= 0.25

        out = optimize_energy(optimizer, walkers_tf, [cur_lr, cur_reg, cur_clip])
        loss = out[-2].numpy[0]
        Emean = out[-1].numpy[0]
        optimizer.update_parameters(out[:-2])
        old_smoothed_energy = status.smoothed_energy
        status.smoothed_energy = smoothing * status.smoothed_energy + (1.0 - smoothing) * Emean
        status.smoothed_loss = smoothing * status.smoothed_loss + (1.0 - smoothing) * loss

        improvement = old_smoothed_energy - status.smoothed_energy
        omsmoothing = 5.0*(1.0 - smoothing)
        status.improvement = (1.0 - omsmoothing) * status.improvement + omsmoothing * improvement

        status.energy_history.append(status.smoothed_energy)

    optimize_energy_time = time.time() - start_time

    return loss, Emean, walkers_tf, metropolis_step_time, optimize_energy_time

walkers = np.random.randn(n_walkers, wavefunction.electron_n, 3).astype(np.float32)
walkers_tf = tf.tensor(walkers)

walker_history = np.random.randn(n_walkers * metropolis_history_length, wavefunction.electron_n, 3).astype(np.float32)
walker_history_tf = tf.tensor(walker_history)


vis = CompileVisualizer(1920, 1080, 1.0)
cam = Camera(position=np.array([0.0, 0.0, 0.0]), quaternion=np.array([0.0, 0.0, 0.0, 1.0]), W=1920, H=1080, angular_speed = 0.005, camera_speed = 0.2)

cam.initialize_parameters()

tf.show_window(cam.W, cam.H, "Walker renderer")

cur_time = time.time()
prev_time = time.time()
smooth_delta_time = 0.0
frame = 0
while not tf.window_should_close():
    cur_time = time.time()
    delta_time = cur_time - prev_time
    smooth_delta_time = 0.9 * smooth_delta_time + 0.1 * delta_time

    loss, Emean, walkers_tf, metropolis_step_time, optimize_energy_time = OptimizationStep(frame, optimizer, walkers_tf, walker_history_tf, status)

    cam.update()

    tf.imgui_text("Simulation time: %.3f ms" % (1000.0 * smooth_delta_time))
    tf.imgui_text("FPS: %.1f" % (1.0 / (smooth_delta_time + 1e-5)))
    
    tf.imgui_text("Energy: %.3f Ha" % status.smoothed_energy)
    tf.imgui_text("Loss: %.3f" % status.smoothed_loss)
    tf.imgui_text("Acceptance rate: %.3f" % status.smoothed_acceptance_rate)
    tf.imgui_text("Improvement: %.3f" % status.improvement)
    tf.imgui_text("Step: %d" % frame)
 
    tf.imgui_text("Error rel: {:.3f} %".format(100.0 * np.abs((status.smoothed_energy - target_energy) / target_energy)))
    tf.imgui_text("Error abs: %.3f mHa" % (1000.0 * np.abs(status.smoothed_energy - target_energy)))

    lr, reg, clip = scheduler(frame)
    tf.imgui_text("Learning rate: %.5f" % lr)
    tf.imgui_text("L2 Regularization: %.5f" % reg)
    tf.imgui_text("Gradient clipping: %.5f" % clip)

    tf.imgui_text("Optimization step took: %.3f ms" % (optimize_energy_time * 1000.0))
    tf.imgui_text("Metropolis steps took: %.3f ms" % (metropolis_step_time * 1000.0))

    #print molecule info
    tf.imgui_text(molecule_info)

    cam.angular_speed = tf.imgui_slider("Angular speed", cam.angular_speed, 0.0, 0.01)
    cam.camera_speed = tf.imgui_slider("Camera speed", cam.camera_speed, 0.0, 0.5)
    cam.focal_length = tf.imgui_slider("Focal length", cam.focal_length, 0.1, 10.0)
    cam.brightness = tf.imgui_slider("Brightness", cam.brightness, 0.0, 5.0)
    cam.distance_clip = tf.imgui_slider("Distance clip", cam.distance_clip, 0.0, 100.0)
    cam.point_radius = tf.imgui_slider("Point radius", cam.point_radius, 0.0, 10.0)

    pause_training = tf.imgui_checkbox("Pause training", pause_training)
    custom_hyperparameters = tf.imgui_checkbox("Custom hyperparameters", custom_hyperparameters)
    custom_lr = tf.imgui_slider("Learning rate", custom_lr, 0.0, 0.01)
    custom_reg = tf.imgui_slider("L2 Regularization", custom_reg, 0.0, 0.01)
    custom_clip = tf.imgui_slider("Gradient clipping", custom_clip, 0.0, 0.5)

    target_acceptance_rate = tf.imgui_slider("Target acceptance rate", target_acceptance_rate, 0.0, 1.0)
    smoothing = tf.imgui_slider("Statistics smoothing", smoothing, 0.9, 1.0)

    tf.imgui_plotlines("Energy history", status.energy_history, graph_size=(0, 200))

    tf.render_frame(vis(cam, walker_history_tf, optimizer.net.atoms))
    frame += 1
    prev_time = cur_time

#print walker position variance
walkers = walkers_tf.numpy
print("Walker position variance: ", np.var(walkers))

#print norm of the gradients
momentum_grads = optimizer.m.items()
for i, grad in momentum_grads:
    print("Parameter ", i, " gradient norm: ", np.linalg.norm(grad.numpy))

#print total trained parameter count (just count all elements in all gradients)
total_param_count = 0
for i, grad in momentum_grads:
    total_param_count += grad.numpy.size
print("Total parameter count: ", total_param_count)