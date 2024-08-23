import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from tqdm import tqdm

from utils import *
from atom import *
from logdet import *
from molecules import *
from visualizer import *

tf.initialize(tf.opengl)

register_logdet()

lr0 = 0.005
lr1 = 0.001
reg0 = 0.05
reg1 = 0.005
n_walkers = 512
opt_steps = 10000

def scheduler(step):
    t = min(max(float(step) / float(opt_steps), 0.0), 1.0)
    return lr0 + (lr1 - lr0) * t,  reg0 + (reg1 - reg0) * t

#how many metropolis steps to perform per optimization step
metropolis_per_step = 12

#what target fraction of the walkers to move in each step
target_acceptance_rate = 0.4

#what fraction of the walkers [sorted by local energy] to ignore in the loss function
#outliers on the tails of the distribution can cause the optimization to diverge due to numerical instability
#ferminet/psiformer used a clipping around the median of the local energy
#but I found that sorting and ignoring a fixed fraction of the sorted walkers is simpler and seems to work well
outlier_fraction = 0.15

smoothing = 0.995

molecule = c2h4_molecule

molecule.print_summary()

target_energy = molecule.target_energy
atoms = molecule.get_atoms()
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
        self.orb_per_atom = 12
        self.determinants = 1
        self.orbital_n = self.orb_per_atom * self.atom_n
        self.mid_n = 24

        # Parameters
        self.params = tf.Parameter([2], tf.float32, requires_grad = False)
        self.step = tf.Parameter([1], tf.int32, requires_grad = False)
        self.seed = tf.Parameter([1], tf.uint32, requires_grad = False)
        self.atoms = tf.Parameter([self.atom_n, 4], tf.float32, requires_grad = False)
        self.probability = tf.Parameter([n_walkers], tf.float32, requires_grad = False)

        # Weights
        self.orbi_layer0 = tf.Parameter([4, self.orb_per_atom], tf.float32)
        self.orbi_layer0_bias = tf.Parameter([self.orb_per_atom], tf.float32)
        self.envelope_layer = tf.Parameter([self.orb_per_atom], tf.float32)

        self.mid_layer1A = tf.Parameter([self.orbital_n, self.mid_n], tf.float32)
        self.mid_layer1B = tf.Parameter([self.orbital_n, self.mid_n], tf.float32)
        self.mid_layer2 = tf.Parameter([self.mid_n, self.mid_n], tf.float32)


        self.exchange_layer_out_up = tf.Parameter([self.mid_n, self.spin_up_n], tf.float32)
        self.exchange_layer_out_down = tf.Parameter([self.mid_n, self.spin_down_n], tf.float32)
        self.exchange_layer_in_up = tf.Parameter([self.spin_up_n, self.mid_n], tf.float32)
        self.exchange_layer_in_down = tf.Parameter([self.spin_down_n, self.mid_n], tf.float32)

        self.up_layer = tf.Parameter([self.mid_n, self.determinants * self.spin_up_n], tf.float32)
        self.down_layer = tf.Parameter([self.mid_n, self.determinants * self.spin_down_n], tf.float32)
        self.gamma = tf.Parameter([4], tf.float32)
        self.dx = 5e-3
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

    def eye(self, N):
        i, j = tf.indices([N, N])
        return tf.select(i == j, 1.0, 0.0)

    def eye_like(self, X):
        i, j = X.indices
        return tf.select(i == j, 1.0, 0.0)
    
    def merge_orbitals(self, data):
        return tf.reshape(data, [data.shape[0], self.electron_n, self.atom_n * self.orb_per_atom])

    def get_spin_up(self, data):
        upi = tf.indices([data.shape[0], self.spin_up_n, data.shape[2]])
        return data[upi[0], upi[1], upi[2]]
    
    def get_spin_down(self, data):
        downi = tf.indices([data.shape[0], self.spin_down_n, data.shape[2]])
        return data[downi[0], self.spin_up_n + downi[1], downi[2]]

    def psi_new(self, electrons):
        tf.region_begin('Psi')

        tf.region_begin('AtomOrbitals')
        #compute electron-atom relative positions and distances [batch, electron, atom, 3] + [batch, electron, atom]
        b, e, a, d = tf.indices([electrons.shape[0], self.electron_n, self.atom_n, 3])
        ri = tf.unsqueeze(electrons, axis=-2) - self.atoms[a, d]
        r = tf.norm(ri)

        #concatenate ri and r [batch, electron, atom, 3 + 1]
        b, e, a, d = tf.indices([electrons.shape[0], self.electron_n, self.atom_n,  4])
        in0 = tf.select(d < 3, ri[b, e, a, d], r[b, e, a]) 

        #orbitals around atoms [batch, electron, atom, orbital]
        out0 = (in0 @ self.orbi_layer0 + self.orbi_layer0_bias)
        atom_orbitals = out0 * tf.exp(-tf.unsqueeze(r) * tf.abs(self.envelope_layer))
        tf.region_end('AtomOrbitals')

        tf.region_begin('Orbitals')
        orbitals = self.merge_orbitals(atom_orbitals)
        orbitals = orbitals @ self.mid_layer1A + tf.tanh(orbitals @ self.mid_layer1B)

        #exchange layer for electron correlation
        exchange_out_up = self.get_spin_up(orbitals) @ self.exchange_layer_out_up
        exchange_out_down = self.get_spin_down(orbitals) @ self.exchange_layer_out_down
        exchange_in_up = tf.unsqueeze(tf.mean(exchange_out_up, axis=1), axis=1)
        exchange_in_down = tf.unsqueeze(tf.mean(exchange_out_down, axis=1), axis=1)
        orbitals = orbitals + (orbitals @ self.mid_layer2) * tf.tanh((exchange_in_up @ self.exchange_layer_in_up) + (exchange_in_down @ self.exchange_layer_in_down))

        #spin up orbitals
        up_features = self.get_spin_up(orbitals) @ self.up_layer
        
        #spin down orbitals
        down_features = self.get_spin_down(orbitals) @ self.down_layer

        tf.region_end('Orbitals')

        tf.region_begin('SlaterDeterminant')

        #compute determinant for each set of electron orbitals [batch, determinant]
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
        x_clipped = tf.clamp(x_sorted, median - 3.0 * mad_median, median + 3.0 * mad_median)

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
    update_prob = tf.input([1], tf.int32)[0]

    new_walkers, acceptance_rate = wavefunction.metropolis_step(walkers, update_prob)

    params = wavefunction.parameters()
    params.append(new_walkers)
    params.append(acceptance_rate)
    return params

metropolis_step = tf.compile(MetropolisStep)

def GetModelOptimizer():
    wavefunction = PSI()
    optimizer = tf.optimizers.adam(wavefunction, beta1 = 0.0, beta2 = 0.95, reg_type = tf.regularizers.l2, reg = 0.02, clip = 0.25)
    return optimizer, wavefunction

def OptimizeEnergy():
    optimizer, wavefunction = GetModelOptimizer()
    optimizer.initialize_input()

    walkers = tf.input([n_walkers, wavefunction.electron_n, 3], tf.float32)
    in_params = tf.input([2], tf.float32)
    optimizer.learning_rate = in_params[0]
    optimizer.reg = in_params[1]
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
optimizer.net.step = tf.tensor(np.array([0], np.int32))
optimizer.net.seed = tf.tensor(np.array([0], np.uint32))

walkers = np.random.randn(n_walkers, wavefunction.electron_n, 3).astype(np.float32)
walkers_tf = tf.tensor(walkers)

acceptance_rate_history = []
energy_history = []


progress_bar = tqdm(range(opt_steps))
smoothed_loss = 0.0
smoothed_energy = 0.0
smoothed_acceptance_rate = 0.0

def OptimizationStep(i):
    for j in range(metropolis_per_step):
        update_prob = [1 if (j == 0) else 0] #must update the stored probability in the first step, since wavefunction has changed
        out = metropolis_step(wavefunction, walkers_tf, update_prob)
        walkers_tf = out[-2]
        acceptance_rate = out[-1]
        wavefunction.update_parameters(out[:-2])
        
        acceptance_rate = acceptance_rate.numpy
        smoothed_acceptance_rate = smoothing * smoothed_acceptance_rate + (1.0 - smoothing) * acceptance_rate[0]
        acceptance_rate_history.append(smoothed_acceptance_rate)
        cur_params = wavefunction.params.numpy

        #update the metropolis step size based on the acceptance rate
        if(acceptance_rate[0] < target_acceptance_rate):
            cur_params[0] *= 0.98
        if(acceptance_rate[0] > target_acceptance_rate):
            cur_params[0] *= 1.02

        wavefunction.params = tf.tensor(cur_params)

    cur_lr, cur_reg = scheduler(i)

    out = optimize_energy(optimizer, walkers_tf, [cur_lr, cur_reg])
    loss = out[-2].numpy
    Emean = out[-1].numpy
    optimizer.update_parameters(out[:-2])

    return loss, Emean

# for i in progress_bar:
#     if(i == 0): tf.renderdoc_start_capture()

#     loss, Emean = OptimizationStep(i)

#     smoothed_energy = smoothing * smoothed_energy + (1.0 - smoothing) * Emean
#     smoothed_loss = smoothing * smoothed_loss + (1.0 - smoothing) * loss
#     energy_history.append(smoothed_energy)

#     progress_bar.set_postfix(acceptance_rate = '{0:.6f}'.format(smoothed_acceptance_rate), loss = '{0:.6f}'.format(smoothed_loss[0]), energy = '{0:.6f}'.format(smoothed_energy[0]))
#     if(i == 0): tf.renderdoc_end_capture()

vis = CompileVisualizer(1280, 720, 1.0)

cam = Camera(position=np.array([0.0, 0.0, 0.0]), quaternion=np.array([0.0, 0.0, 0.0, 1.0]), W=1280, H=720, focal_length=1.0, angular_speed = 0.005, camera_speed = 0.01)
cam.initialize_parameters()

tf.show_window(cam.W, cam.H, "Walker renderer")

frame = 0
while not tf.window_should_close():
    cam.controller_update()
    OptimizationStep(frame)
    image = vis(cam, walkers_tf)
    tf.render_frame(image)
    frame += 1

print("Final Energy: ", np.mean(smoothed_energy))
print("Error, %: ", 100.0 * np.abs((np.mean(smoothed_energy) - target_energy) / target_energy))
#print weights
#print("Weights: ", wavefunction.weights.numpy)

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