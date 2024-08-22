import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from tqdm import tqdm

from utils import *
from atom import *
from logdet import *

tf.initialize(tf.opengl)

register_logdet()

lr = 0.005
n_walkers = 2048
opt_steps = 5000

#how many metropolis steps to perform per optimization step
metropolis_per_step = 12

#what target fraction of the walkers to move in each step
target_acceptance_rate = 0.5

#what fraction of the walkers [sorted by local energy] to ignore in the loss function
#outliers on the tails of the distribution can cause the optimization to diverge due to numerical instability
#ferminet/psiformer used a clipping around the median of the local energy
#but I found that sorting and ignoring a fixed fraction of the sorted walkers is simpler and seems to work well
outlier_fraction = 0.25


smoothing = 0.96

# atom1 = Atom(3, "Li", Vector3(-5.051*0.5, 0.0, 0.0))
# atom2 = Atom(3, "Li", Vector3(5.051*0.5, 0.0, 0.0))
# molecule = Molecule([atom1, atom2])
# molecule.initialize_orbitals()
# target_energy = -14.9954

# atom = Atom(6, "C", Vector3(0.0, 0.0, 0.0))
# molecule = Molecule([atom])
# molecule.initialize_orbitals()
# target_energy = -37.846772

# Methane
# atomC = Atom(6, "C", Vector3(0.0, 0.0, 0.0))
# atomH1 = Atom(1, "H", Vector3(1.18886, 1.18886, 1.18886))
# atomH2 = Atom(1, "H", Vector3(-1.18886, -1.18886, 1.18886))
# atomH3 = Atom(1, "H", Vector3(-1.18886, 1.18886, -1.18886))
# atomH4 = Atom(1, "H", Vector3(1.18886, -1.18886, -1.18886))
# molecule = Molecule([atomC, atomH1, atomH2, atomH3, atomH4])
# molecule.initialize_orbitals()
# target_energy = -40.51400

# Oxygen
# atom = Atom(8, "O", Vector3(0.0, 0.0, 0.0))
# molecule = Molecule([atom])
# molecule.initialize_orbitals()
# target_energy = -75.06655

# Neon
# atom = Atom(10, "Ne", Vector3(0.0, 0.0, 0.0))
# molecule = Molecule([atom])
# molecule.initialize_orbitals()
# target_energy = -128.9366

# Ethane
# C1 (0.0, 0.0, 1.26135)
# C2 (0.0, 0.0, -1.26135)
# H1 (0.0, 1.74390, 2.33889)
# H2 (0.0, -1.74390, 2.33889)
# H3 (0.0, 1.74390, -2.33889)
# H4 (0.0, -1.74390, -2.33889)
atomC1 = Atom(6, "C", Vector3(0.0, 0.0, 1.26135))
atomC2 = Atom(6, "C", Vector3(0.0, 0.0, -1.26135))
atomH1 = Atom(1, "H", Vector3(0.0, 1.74390, 2.33889))
atomH2 = Atom(1, "H", Vector3(0.0, -1.74390, 2.33889))
atomH3 = Atom(1, "H", Vector3(0.0, 1.74390, -2.33889))
atomH4 = Atom(1, "H", Vector3(0.0, -1.74390, -2.33889))
molecule = Molecule([atomC1, atomC2, atomH1, atomH2, atomH3, atomH4])
molecule.initialize_orbitals()
target_energy = -78.5844


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
        self.mid_n = 16

        self.params = tf.Parameter([2], tf.float32, requires_grad = False)
        self.step = tf.Parameter([1], tf.int32, requires_grad = False)
        self.seed = tf.Parameter([1], tf.uint32, requires_grad = False)
        self.atoms = tf.Parameter([self.atom_n, 4], tf.float32, requires_grad = False)
        self.orbi_layer0 = tf.Parameter([4, self.orb_per_atom], tf.float32)
        self.orbi_layer0_bias = tf.Parameter([self.orb_per_atom], tf.float32)
        self.envelope_layer = tf.Parameter([self.orb_per_atom], tf.float32)
        self.mid_layer1A = tf.Parameter([self.orbital_n, self.mid_n], tf.float32)
        self.mid_layer1B = tf.Parameter([self.orbital_n, self.mid_n], tf.float32)
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

        envelope = out0 * tf.exp(-tf.unsqueeze(r) * tf.abs(self.envelope_layer))

        tf.region_end('AtomOrbitals')

        tf.region_begin('Orbitals')
        midi = tf.indices([electrons.shape[0], self.electron_n, self.atom_n * self.orb_per_atom])
        envelope = envelope[midi[0], midi[1], midi[2] / self.orb_per_atom, midi[2] % self.orb_per_atom]

        envelope = envelope @ self.mid_layer1A + tf.tanh(envelope @ self.mid_layer1B)

        #reshape, merge atom orbitals into one dimension [batch, electrons, atom*orbital]
        up1i = tf.indices([electrons.shape[0], self.spin_up_n, self.mid_n])
        up1 = envelope[up1i[0], up1i[1], up1i[2]]
        up_features = up1 @ self.up_layer
        
        down1i = tf.indices([electrons.shape[0], self.spin_down_n, self.mid_n])
        down1 = envelope[down1i[0], self.spin_up_n + down1i[1], down1i[2]]
        down_features = down1 @ self.down_layer

        tf.region_end('Orbitals')

        tf.region_begin('SlaterDeterminant')
        det_down = self.logdeterminant(down_features, self.spin_down_n)
        det_up = self.logdeterminant(up_features, self.spin_up_n)

        #compute determinant for each set of electron orbitals [batch, determinant]
        #orbiprod = det_up * det_down
        #orbiprod = tf.squeeze(det_up) + tf.squeeze(det_down)
        orbiprod = det_up + det_down
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
        # b, d = tf.indices([orbitals.shape[0], self.determinants])

        # m = []
        # for i in range(N):
        #     m.append([])
        #     for j in range(N):
        #         m[i].append(aslog(orbitals[b, i, j, d]))

        # return self.logdeterminant_laplace(m, N)

        bi, i, j = tf.indices([orbitals.shape[0] * self.determinants, N, N])
        b = bi / self.determinants
        d = bi % self.determinants

        reordered = orbitals[b, i, j, d]
        logdet = tf.custom("logdet", [reordered], [reordered.shape[0]])
        logdet = tf.reshape(logdet, [orbitals.shape[0], self.determinants])
        logdet = tf.squeeze(logdet)
        return logdet / 0.69314718056
        return tf.squeeze(logdet / 0.69314718056)


    #VERY slow determinant calculation using Laplace expansion
    def logdeterminant_laplace(self, m, N):
        if N <= 0:
            return lognum(0.0, 1.0)

        if N == 1:
            return m[0][0]
        
        if N == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        
        result = aslog(0.0)
        for j in range(N):
            submatrix = [m[i][:j] + m[i][j + 1:] for i in range(1, N)]
            sign = lognum(0.0, 1.0 if j % 2 == 0 else -1.0)
            term = sign * m[0][j] * self.logdeterminant_laplace(submatrix, N - 1)
            result = term + result
        return result
    
    #unrolled determinant calculation (not exactly the best way to do this, but easy auto-differentiation)
    def determinant(self, orbitals, N):
        orbitals = tf.reshape(orbitals, [orbitals.shape[0], N, N, self.determinants])
        b, d = tf.indices([orbitals.shape[0], self.determinants])

        m = []
        for i in range(N):
            m.append([])
            for j in range(N):
                m[i].append(orbitals[b, i, j, d])

        return aslog(self.determinant_laplace(m, N))

    #VERY slow determinant calculation using Laplace expansion
    def determinant_laplace(self, m, N):
        if N == 0:
            return 1.0

        if N == 1:
            return m[0][0]

        if N == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        
        if N > 2:
            result = 0.0
            for j in range(N):
                submatrix = [m[i][:j] + m[i][j + 1:] for i in range(1, N)]
                sign = 1.0 if j % 2 == 0 else -1.0
                term = sign * m[0][j] * self.determinant_laplace(submatrix, N - 1)
                result += term
        return result

    #computing psi in log space is more numerically stable
    def log_psi(self, electrons):
        #psi = self.chandrasekhar_helium_psi(electrons)
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

    def prob_density(self, e_pos):
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


    def metropolis_step(self, e_pos):
        old_prob = self.prob_density(e_pos)

        e_pos_new = e_pos + self.randn(e_pos.shape) * self.metropolis_dx()
        new_prob = self.prob_density(e_pos_new)

        ratio = tf.exp(tf.clamp(new_prob - old_prob, -50.0, 50.0))

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

def GetModelOptimizer():
    wavefunction = PSI()
    optimizer = tf.optimizers.adam(wavefunction, learning_rate = lr, beta1 = 0.0, reg_type = tf.regularizers.l2, reg = 0.05, clip = 0.5)
    return optimizer, wavefunction

def OptimizeEnergy():
    optimizer, wavefunction = GetModelOptimizer()
    optimizer.initialize_input()

    walkers = tf.input([n_walkers, wavefunction.electron_n, 3], tf.float32)
    loss, Emean = wavefunction.loss(walkers)
    optimizer.step(loss)
    params = optimizer.parameters()
    params.append(loss)
    params.append(Emean)
    return params

optimize_energy = tf.compile(OptimizeEnergy)

np.random.seed(0)

optimizer, wavefunction = GetModelOptimizer()
optimizer.initialize_parameters()
optimizer.net.params = tf.tensor(np.array([0.5, outlier_fraction]).astype(np.float32))
#optimizer.net.weights = tf.tensor(np.array([1.5185, 2.2154, 0.3604]))
optimizer.net.atoms = tf.tensor(atoms)
optimizer.net.step = tf.tensor(np.array([0], np.int32))
optimizer.net.seed = tf.tensor(np.array([0], np.uint32))

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
for i in progress_bar:
    if(i == 100): tf.renderdoc_start_capture()

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

        wavefunction.params = tf.tensor(cur_params)

    out = optimize_energy(optimizer, walkers_tf)
    loss = out[-2].numpy
    Emean = out[-1].numpy
    optimizer.update_parameters(out[:-2])

    smoothed_energy = smoothing * smoothed_energy + (1.0 - smoothing) * Emean
    smoothed_loss = smoothing * smoothed_loss + (1.0 - smoothing) * loss
    energy_history.append(smoothed_energy)

    progress_bar.set_postfix(acceptance_rate = '{0:.6f}'.format(smoothed_acceptance_rate), loss = '{0:.6f}'.format(smoothed_loss[0]), energy = '{0:.6f}'.format(smoothed_energy[0]))
    if(i == 100): tf.renderdoc_end_capture()

print("Final Energy: ", np.mean(smoothed_energy))
print("Error, %: ", 100.0 * np.abs((np.mean(smoothed_energy) - target_energy) / target_energy))
#print weights
#print("Weights: ", wavefunction.weights.numpy)

#plot energy history
plt.figure()
plt.plot(energy_history)
plt.xlabel('Step')
plt.ylabel('Energy, Hartree')
plt.title('Energy History')
plt.grid(True)
plt.show()

#plot walker scatter plot in XY plane for all electrons
walkers = walkers_tf.numpy
plt.figure()
Xwalkers = walkers[:, :, 0].flatten()
Ywalkers = walkers[:, :, 1].flatten()
plt.scatter(Xwalkers, Ywalkers, s = 5, alpha = 0.05)
#plot atoms and their names (white text with black border)
for i in range(atoms.shape[0]):
    plt.text(atoms[i, 0], atoms[i, 1], molecule.atoms[i].name, color = 'white', fontsize = 20, va = 'center', ha = 'center', path_effects = [patheffects.withStroke(linewidth = 2, foreground = 'black')])
    
plt.xlabel('X, Bohr')
plt.ylabel('Y, Bohr')
plt.title('Walker Scatter Plot')
plt.xlim(-5.0, 5.0)
plt.ylim(-5.0, 5.0)
plt.grid(True)
plt.show()

# #print out parameters
# print("envelope_layer: ", wavefunction.envelope_layer.numpy)
# print("orbi_layer0: ", wavefunction.orbi_layer0.numpy)
# print("orbi_layer0_bias: ", wavefunction.orbi_layer0_bias.numpy)
# print("orbi_layer1: ", wavefunction.orbi_layer1.numpy)
# print("orbi_layer1_bias: ", wavefunction.orbi_layer1_bias.numpy)
# print("orbi_layer2: ", wavefunction.orbi_layer2.numpy)
# print("gamma: ", wavefunction.gamma.numpy)
# print("gamma2: ", wavefunction.gamma2.numpy)
# print("dx: ", wavefunction.dx)


#print norm of the gradients
momentum_grads = optimizer.m.items()
for i, grad in momentum_grads:
    print("Parameter ", i, " gradient norm: ", np.linalg.norm(grad.numpy))