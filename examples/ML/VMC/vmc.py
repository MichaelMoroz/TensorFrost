import TensorFrost as tf
from TensorFrost import imgui
from TensorFrost import window
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

molecule = ch4_molecule

lr0 = 0.004
lr1 = 0.0005
reg0 = 0.01
reg1 = 0.00
clip0 = 0.005
clip1 = 0.005
n_walkers = 1024
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
    return cur_lr, cur_reg, cur_clip

#how many metropolis steps to perform per optimization step
metropolis_per_step = 12

#how many metropolis steps to store in the history to render
metropolis_history_length = 32

#what target fraction of the walkers to move in each step
target_acceptance_rate = 0.25

#what fraction of the walkers [sorted by local energy] to ignore in the loss function
#outliers on the tails of the distribution can cause the optimization to diverge due to numerical instability
#but I found that sorting and ignoring a fixed fraction of the sorted walkers is simple and works well
outlier_fraction = 0.15
#range for clipping around the median of the local energy from ferminet/psiformer (in variance units)
mad_range = 5.0

#how many previous steps to average the local energy statistics
average_count = 1024

#finite difference step factor for computing the laplacian
#(currently is modulated by the distance to the nearest atom)
#(will be replaced by forward mode autodiff in the future)
finitedifference_dx = 1.5e-2

#get the molecule information
molecule_info = molecule.get_summary()
target_energy = molecule.target_energy
atoms = molecule.get_atoms()
feature_atom_id = molecule.get_orbitals(2, 3)
print(feature_atom_id)
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
        self.atom_features = feature_atom_id.shape[0]
        self.mid_n = 24
        self.corr_n = 12
        self.input_features = 8

        # Hyperparameters
        self.params = tf.Parameter([-1], tf.float32, requires_grad = False)
        self.step = tf.Parameter([1], tf.int32, requires_grad = False)
        self.seed = tf.Parameter([1], tf.uint32, requires_grad = False)
        self.atoms = tf.Parameter([self.atom_n, 4], tf.float32, requires_grad = False)
        self.feature_atom_id = tf.Parameter([self.atom_features], tf.int32, requires_grad = False)
        self.probability = tf.Parameter([n_walkers], tf.float32, requires_grad = False)

        # Weights
        self.feature_layer = tf.Parameter([self.atom_features, self.input_features], tf.float32, random_scale = 0.1)
        self.feature_layer_bias = tf.Parameter([self.atom_features], tf.float32, random_scale = 1.0)
        self.feature_envelope_layer = tf.Parameter([self.atom_features], tf.float32, random_scale = 0.25, random_offset = 1.5)

        self.features_to_orbitals = tf.Parameter([self.atom_features, self.mid_n], tf.float32)
        self.orbi_layer1 = tf.Parameter([self.mid_n, self.mid_n], tf.float32)
        self.orbi_layer1_bias = tf.Parameter([self.mid_n], tf.float32, random_scale = 1.0)
        self.orbi_layer2 = tf.Parameter([self.mid_n, self.mid_n], tf.float32)
        self.orbi_layer3 = tf.Parameter([self.mid_n, self.mid_n], tf.float32)
        self.orbi_layer3_bias = tf.Parameter([self.mid_n], tf.float32, random_scale = 1.0)

        self.exchange_layer_out_up = tf.Parameter([self.mid_n, self.corr_n], tf.float32)
        self.exchange_layer_out_down = tf.Parameter([self.mid_n, self.corr_n], tf.float32)
        self.exchange_layer_in_up = tf.Parameter([self.corr_n, self.mid_n], tf.float32)
        self.exchange_layer_in_down = tf.Parameter([self.corr_n, self.mid_n], tf.float32)

        self.up_feature_layer = tf.Parameter([self.mid_n, self.determinants * self.spin_up_n], tf.float32)
        self.down_feature_layer = tf.Parameter([self.mid_n, self.determinants * self.spin_down_n], tf.float32)
        self.gamma = tf.Parameter([4], tf.float32)
        self.dx = 5e-3
        self.eps = 1e-7

    def metropolis_dx(self):
        return self.params[0]
    
    def outlier_fraction(self):
        return self.params[1]
    
    def mad_range(self):
        return self.params[2]
    
    def finitedifference_dx(self):
        return self.params[3]

    def assert_parameters(self):
        self.seed = tf.assert_tensor(self.seed, [1], tf.uint32)

    def get_spin_up(self, data):
        upi = tf.indices([data.shape[0], self.spin_up_n, data.shape[2]])
        return data[upi[0], upi[1], upi[2]]
    
    def get_spin_down(self, data):
        downi = tf.indices([data.shape[0], self.spin_down_n, data.shape[2]])
        return data[downi[0], self.spin_up_n + downi[1], downi[2]]

    def psi_new(self, electrons):
        tf.region_begin('Psi')

        tf.region_begin('AtomFeatures')
        #compute electron-atom relative positions and distances [batch, electron, orbital_features, 3] + [batch, electron, orbital_features]
        b, e, a, d = tf.indices([electrons.shape[0], self.electron_n, self.atom_features, 3])
        atom_id = self.feature_atom_id[a]
        ri = tf.unsqueeze(electrons, axis=-2) - self.atoms[atom_id, d]
        r = tf.norm(ri)

        #concatenate input features
        b, e, a, d = tf.indices([electrons.shape[0], self.electron_n, self.atom_features, self.input_features])
        in0 = tf.select(d < 3, ri[b, e, a, d], r[b, e, a]) #concatenate ri and r
        in0 = tf.select(d < 4, in0, 0.5 * ri[b, e, a, d-4] ** 2.0) #add ri^2
        in0 = tf.select(d < 7, in0, 0.5 * tf.float(e < self.spin_up_n)) #add spin up/down indicator
 
        #atom featuress
        out0 = (tf.dot(in0, self.feature_layer) + self.feature_layer_bias)
        atom_features = out0 * tf.exp(-r * tf.abs(self.feature_envelope_layer))
        tf.region_end('AtomFeatures')

        tf.region_begin('Orbitals')
        orbitals = atom_features @ self.features_to_orbitals 
        orbitals = orbitals * tf.tanh(self.orbi_layer1_bias + orbitals @ self.orbi_layer1)

        #exchange layer for symmetric information exchange between electrons
        exchange_out_up = self.get_spin_up(orbitals) @ self.exchange_layer_out_up
        exchange_out_down = self.get_spin_down(orbitals) @ self.exchange_layer_out_down
        #average the exchange features to make it symmetric (could also use attention)
        exchange_in_up = tf.unsqueeze(tf.mean(exchange_out_up, axis=1), axis=1)
        exchange_in_down = tf.unsqueeze(tf.mean(exchange_out_down, axis=1), axis=1)
        exchange_features_in = exchange_in_up @ self.exchange_layer_in_up + exchange_in_down @ self.exchange_layer_in_down
        orbitals = orbitals + tf.tanh((orbitals @ self.orbi_layer2) * exchange_features_in)

        orbitals = orbitals * tf.tanh(self.orbi_layer3_bias + orbitals @ self.orbi_layer3)

        #up and down features
        #note: since we didn't use biases and multiplied the exchange layer with the orbitals, we dont really need an envelope function here
        #the orbitals should already satisfy the boundary conditions
        up_features = self.get_spin_up(orbitals) @ self.up_feature_layer
        down_features = self.get_spin_down(orbitals) @ self.down_feature_layer
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
        #TODO: fix compiler bug here
        return tf.squeeze(logdet / 0.69314718056)

    #computing psi in log space is more numerically stable
    def log_psi(self, electrons):
        psi = self.psi_new(electrons)
        return psi.log()

    #get the finite difference gradient and laplacian
    def kinetic_energy(self, e_pos):
        logpsi_center = self.log_psi(e_pos)

        tf.region_begin('Laplacian')

        #compute distance to atoms for better numerical stability when computing the finite difference
        cd = self.min_distance_to_atoms(e_pos)
        dx_estimate = tf.max(cd * self.finitedifference_dx(), 5e-3)

        #fd sampling (TODO: use forward mode autodiff)
        n_samples = self.electron_n * 3 * 2
        b, s, i, c = tf.indices([e_pos.shape[0], n_samples, self.electron_n, 3])
        eid = s / 6 #electron index
        did = (s / 2) % 3 #dimension index
        sid = s % 2 #forward or backward
        cd_dx = dx_estimate[b, i]
        deltaD = tf.select(sid == 0, cd_dx, -cd_dx)
        pos = tf.unsqueeze(e_pos, axis=1) + tf.select((did == c) & (eid == i), deltaD, 0.0)
        pos = tf.reshape(pos, [e_pos.shape[0] * n_samples, self.electron_n, 3])
        logpsi = self.log_psi(pos)
        logpsi = tf.reshape(logpsi, [e_pos.shape[0], n_samples])

        #laplacian of the wavefunction divided by the wavefunction (which is the laplacian + the gradient squared of the log wavefunction)
        b, = tf.indices([e_pos.shape[0]])
        kinetic = tf.const(0.0)

        with tf.loop(self.electron_n) as electron:
            with tf.loop(3) as d:
                psi0 = logpsi[b, 6 * electron + 2 * d + 0]
                psi1 = logpsi[b, 6 * electron + 2 * d + 1]
                this_dx = dx_estimate[b, electron]
                kinetic.val += (psi0 + psi1 - 2.0 * logpsi_center + 0.25*(psi0 - psi1)*(psi0 - psi1)) / (this_dx * this_dx)

        tf.region_end('Laplacian')
        return - 0.5 * kinetic, logpsi_center
    
    def electron_potential(self, e_pos):
        b, = tf.indices([e_pos.shape[0]])
        V = tf.const(0.0)

        #compute the nuclei potential sum for each electron*nuclei
        with tf.loop(self.electron_n) as electron:
            with tf.loop(self.atom_n) as n:
                r = tf.sqrt(sqr(e_pos[b, electron, 0] - self.atoms[n, 0]) + sqr(e_pos[b, electron, 1] - self.atoms[n, 1]) + sqr(e_pos[b, electron, 2] - self.atoms[n, 2]))
                V.val -= self.atoms[n, 3] / tf.max(r, self.eps)

        #compute the electron-electron potential sum
        with tf.loop(self.electron_n) as electron:
            with tf.loop(electron + 1, self.electron_n) as f:
                r = tf.sqrt(sqr(e_pos[b, electron, 0] - e_pos[b, f, 0]) + sqr(e_pos[b, electron, 1] - e_pos[b, f, 1]) + sqr(e_pos[b, electron, 2] - e_pos[b, f, 2]))
                V.val += 1.0 / tf.max(r, self.eps)

        return V

    def nuclei_potential(self):
        V = tf.const(0.0)

        #compute the potential between nuclei
        with tf.loop(self.atom_n) as n:
            with tf.loop(n + 1, self.atom_n) as m:
                r = tf.sqrt(sqr(self.atoms[n, 0] - self.atoms[m, 0]) + sqr(self.atoms[n, 1] - self.atoms[m, 1]) + sqr(self.atoms[n, 2] - self.atoms[m, 2]))
                V.val += self.atoms[n, 3] * self.atoms[m, 3] / tf.max(r, self.eps)

        return V
    
    #average distance to atoms weighted by the atom weights
    def min_distance_to_atoms(self, e_pos):
        b, i = tf.indices([e_pos.shape[0], e_pos.shape[1]])
        cd = tf.const(1e10)
        #norm = tf.const(0.0)
        with tf.loop(self.atom_n) as n:
            d = tf.sqrt(sqr(e_pos[b, i, 0] - self.atoms[n, 0]) + sqr(e_pos[b, i, 1] - self.atoms[n, 1]) + sqr(e_pos[b, i, 2] - self.atoms[n, 2]))
            #w = self.atoms[n, 3]
            cd.val = tf.min(cd, d)
            #norm.val += w
        return cd #/ tf.max(norm, self.eps)
    
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
        cur_mad_range = self.mad_range()
        x_clipped = tf.clamp(x_sorted, median - cur_mad_range * mad_median, median + cur_mad_range * mad_median)

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

        metro_dx = self.metropolis_dx()
        dx_estimate_old = metro_dx*tf.max(self.min_distance_to_atoms(e_pos), 2.5) 

        de_pos = tf.select(update_prob, 0.0, self.randn(e_pos.shape) * tf.unsqueeze(dx_estimate_old))
        #make sure they dont fly away
        e_pos_new = tf.clamp(e_pos + de_pos, -10.0, 10.0)
        new_prob = self.log_prob_density(e_pos_new)

        de_r = tf.norm(de_pos)
        dx_estimate_new = metro_dx*tf.max(self.min_distance_to_atoms(e_pos_new), 2.5)
        proposal_old_to_new = tf.log(self.gaussian_probability_3d(de_r, dx_estimate_old))
        proposal_new_to_old = tf.log(self.gaussian_probability_3d(de_r, dx_estimate_new))

        proposal_ratio = tf.sum(proposal_new_to_old - proposal_old_to_new)

        ratio = tf.exp(tf.clamp(new_prob - old_prob, -50.0, 50.0) + proposal_ratio)

        accept = (self.rand(ratio.shape) < ratio)

        #update the probability
        self.probability = tf.select(accept | update_prob, new_prob, old_prob)

        acceptance_rate = tf.mean(tf.float(accept))
        accept = tf.unsqueeze(tf.unsqueeze(accept))

        #update the electron positions
        e_pos = tf.select(accept & (~update_prob), e_pos_new, e_pos)

        self.inc_step()
        return e_pos, acceptance_rate
    
    def gaussian_probability_3d(self, r, sigma):
        return tf.exp(-0.5 * sqr(r) / sqr(sigma)) / tf.pow(2.0 * np.pi * sqr(sigma), 3.0 / 2.0)

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

np.random.seed(3)

optimizer, wavefunction = GetModelOptimizer()
optimizer.initialize_parameters()
optimizer.net.params = tf.tensor(np.array([0.5, outlier_fraction, mad_range, finitedifference_dx], np.float32))
optimizer.net.atoms = tf.tensor(atoms)
optimizer.net.feature_atom_id = tf.tensor(feature_atom_id)
optimizer.net.step = tf.tensor(np.array([0], np.int32))
optimizer.net.seed = tf.tensor(np.array([0], np.uint32))

class StatisticsLogger:
    def __init__(self):
        self.history = np.array([])
        self.avg_history = np.array([])
        self.value = 0.0
        self.avg_value = 0.0
        self.improvement = 0.0
        self.variance = 0.0
    
    def update(self, value):
        self.value = value
        self.history = np.append(self.history, value)
        chunk = self.history[-average_count:]
        self.avg_value = np.mean(chunk)
        self.variance = np.var(chunk)
        self.avg_history = np.append(self.avg_history, self.avg_value)
        avg_chunk = self.avg_history[-average_count:]
        self.improvement = -np.mean(np.diff(avg_chunk))

energy_log = StatisticsLogger()
loss_log = StatisticsLogger()
acceptance_rate_log = StatisticsLogger()

def OptimizationStep(i, optimizer, walkers_tf, walker_history_tf):

    start_time = time.time()

    for j in range(metropolis_per_step):
        update_prob = 1 if (j == 0) else 0 #must update the stored probability in the first step, since wavefunction has changed
        total_step = i * metropolis_per_step + j
        history_step = total_step % metropolis_history_length
        metropolis_params = [update_prob, history_step]
        out = metropolis_step(optimizer.net, walkers_tf, walker_history_tf, metropolis_params)
        walkers_tf = out[-2]
        acceptance_rate = out[-1].numpy
        optimizer.net.update_parameters(out[:-2])
        cur_params = optimizer.net.params.numpy
        
        acceptance_rate_log.update(acceptance_rate[0])
        #update the metropolis step size based on the acceptance rate
        if(acceptance_rate[0] < target_acceptance_rate):
            cur_params[0] *= 0.98
        if(acceptance_rate[0] > target_acceptance_rate):
            cur_params[0] *= 1.02

        cur_params[1] = outlier_fraction
        cur_params[2] = mad_range
        cur_params[3] = finitedifference_dx

        optimizer.net.params = tf.tensor(cur_params)

    metropolis_step_time = time.time() - start_time
    start_time = time.time()

    cur_lr, cur_reg, cur_clip = scheduler(i)

    loss, Emean = 0.0, 0.0
    if not pause_training:
        if(energy_log.improvement < 0.0):
            cur_lr *= 0.25

        out = optimize_energy(optimizer, walkers_tf, [cur_lr, cur_reg, cur_clip])
        loss_log.update(out[-2].numpy[0])
        energy_log.update(out[-1].numpy[0])
        optimizer.update_parameters(out[:-2])

    optimize_energy_time = time.time() - start_time

    return loss, Emean, walkers_tf, metropolis_step_time, optimize_energy_time

walkers = np.random.randn(n_walkers, wavefunction.electron_n, 3).astype(np.float32)
walkers_tf = tf.tensor(walkers)

walker_history = np.random.randn(n_walkers * metropolis_history_length, wavefunction.electron_n, 3).astype(np.float32)
walker_history_tf = tf.tensor(walker_history)


vis = CompileVisualizer(1920, 1080, 1.0)
cam = Camera(position=np.array([0.0, 0.0, 0.0]), quaternion=np.array([0.0, 0.0, 0.0, 1.0]), W=1920, H=1080, angular_speed = 0.005, camera_speed = 0.2)
cam.brightness = 0.15
cam.initialize_parameters()

window.show(cam.W, cam.H, "Walker renderer")

cur_time = time.time()
prev_time = time.time()
smooth_delta_time = 0.0
frame = 0
while not window.should_close():
    cur_time = time.time()
    delta_time = cur_time - prev_time
    smooth_delta_time = 0.9 * smooth_delta_time + 0.1 * delta_time

    loss, Emean, walkers_tf, metropolis_step_time, optimize_energy_time = OptimizationStep(frame, optimizer, walkers_tf, walker_history_tf)

    imgui.text("Simulation time: %.3f ms" % (1000.0 * smooth_delta_time))
    imgui.text("FPS: %.1f" % (1.0 / (smooth_delta_time + 1e-5)))
    
    imgui.text("Smoothed Energy: %.4f Ha" % energy_log.avg_value)
    imgui.text("Energy statistics variance: %.4f Ha" % energy_log.variance)
    imgui.text("Smoothed loss: %.4f" % loss_log.avg_value)
    imgui.text("Loss: %.3f" % loss_log.value)
    imgui.text("Energy: %.3f" % energy_log.value)
    imgui.text("Acceptance rate: %.3f" % acceptance_rate_log.value)
    imgui.text("Improvement: %.5f" % energy_log.improvement)
    imgui.text("Step: %d" % frame)
 
    imgui.text("Error rel: {:.4f} %".format(100.0 * np.abs((energy_log.avg_value - target_energy) / target_energy)))
    imgui.text("Error abs: %.4f mHa" % (1000.0 * np.abs(energy_log.avg_value - target_energy)))

    lr, reg, clip = scheduler(frame)
    imgui.text("Learning rate: %.5f" % lr)
    imgui.text("L2 Regularization: %.5f" % reg)
    imgui.text("Gradient clipping: %.5f" % clip)

    imgui.text("Optimization step took: %.3f ms" % (optimize_energy_time * 1000.0))
    imgui.text("Metropolis steps took: %.3f ms" % (metropolis_step_time * 1000.0))

    #print molecule info
    imgui.text(molecule_info)

    cam.angular_speed = imgui.slider("Angular speed", cam.angular_speed, 0.0, 0.01)
    cam.camera_speed = imgui.slider("Camera speed", cam.camera_speed, 0.0, 0.5)
    cam.focal_length = imgui.slider("Focal length", cam.focal_length, 0.1, 10.0)
    cam.brightness = imgui.slider("Brightness", cam.brightness, 0.0, 0.5)
    cam.distance_clip = imgui.slider("Distance clip", cam.distance_clip, 0.0, 100.0)
    cam.point_radius = imgui.slider("Point radius", cam.point_radius, 0.0, 10.0)

    pause_training = imgui.checkbox("Pause training", pause_training)
    custom_hyperparameters = imgui.checkbox("Custom hyperparameters", custom_hyperparameters)
    custom_lr = imgui.slider("Learning rate", custom_lr, 0.0, 0.01)
    custom_reg = imgui.slider("L2 Regularization", custom_reg, 0.0, 0.01)
    custom_clip = imgui.slider("Gradient clipping", custom_clip, 0.0, 0.5)

    target_acceptance_rate = imgui.slider("Target acceptance rate", target_acceptance_rate, 0.0, 1.0)
    outlier_fraction = imgui.slider("Outlier fraction", outlier_fraction, 0.0, 0.3)
    mad_range = imgui.slider("MAD clipping range", mad_range, 0.5, 10.0)
    finitedifference_dx = imgui.slider("Finite difference dx", finitedifference_dx, 1e-5, 1e-1)
    average_count = imgui.slider("Average stats over", average_count, 1, 4096)

    imgui.plotlines("Energy history", energy_log.avg_history[-512:], graph_size=(0, 200))

    cam.update()

    atom_screen = ProjectPoints(cam, atoms)
    for i in range(atoms.shape[0]):
        if atom_screen[i, 2] > 0.0:
            q = atoms[i, 3]
            name = Atom.ELEMENT_NAMES[int(q)]
            imgui.add_background_text(name, (atom_screen[i, 1], atom_screen[i, 0]), (255, 255, 255, 255))

    cam.update_tensors()
    window.render_frame(vis(cam, walker_history_tf, optimizer.net.atoms))
    
    frame += 1
    prev_time = cur_time

# #print walker position variance
# walkers = walkers_tf.numpy
# print("Walker position variance: ", np.var(walkers))

# #print norm of the gradients
# momentum_grads = optimizer.m.items()
# for i, grad in momentum_grads:
#     print("Parameter ", i, " gradient norm: ", np.linalg.norm(grad.numpy))

# #print total trained parameter count (just count all elements in all gradients)
# total_param_count = 0
# for i, grad in momentum_grads:
#     total_param_count += grad.numpy.size
# print("Total parameter count: ", total_param_count)

# print(wavefunction.feature_layer.numpy)
# print(wavefunction.feature_layer_bias.numpy)
# print(wavefunction.feature_envelope_layer.numpy)