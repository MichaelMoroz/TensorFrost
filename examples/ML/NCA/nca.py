import TensorFrost as tf
import numpy as np

CHANNEL_N = 12       # Number of CA state channels
TARGET_PADDING = 8   # Number of pixels used to pad the target image border
TARGET_SIZE = 40
BATCH_SIZE = 3*3
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.75

INFERENCE_SIZE = 384

def GELU(X):
    return 0.5*X*(1.0 + tf.tanh(np.sqrt(2.0/np.pi) * (X + 0.044715 * (X * X * X))))

class CAModel(tf.Module):
    def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.hidden_size = 128

        self.fc1 = tf.Parameter([channel_n * 4, self.hidden_size], tf.float32)
        self.fc1_bias = tf.Parameter([self.hidden_size], tf.float32, random_scale = 0.0)
        self.fc2 = tf.Parameter([self.hidden_size, channel_n], tf.float32, random_scale = 0.0)
        self.fc2_bias = tf.Parameter([channel_n], tf.float32, random_scale = 0.0)

        self.filters = tf.Parameter([3, 3, 3], tf.float32, random_scale = 0.0, optimize = False)
        self.seed = tf.Parameter([1], tf.uint32, random_scale = 0.0, optimize = False)

    def assert_parameters(self):
        self.fc2 = tf.assert_tensor(self.fc2, [self.fc1.shape[1], self.fc2.shape[1]], tf.float32)

    def filter(self, X, W):
        bi, wi, hi, ch, fid, it = tf.indices([X.shape[0], X.shape[1], X.shape[2], X.shape[3], W.shape[0], 9])
        i, j = it % 3, it / 3
        conv = tf.sum(X[bi, wi + i - 1, hi + j - 1, ch] * W[fid, i, j])
        return conv
    
    def max_neighbor_alpha(self, X):
        bi, wi, hi, it = tf.indices([X.shape[0], X.shape[1], X.shape[2], 9])
        i, j = it % 3, it / 3
        return tf.unsqueeze(tf.max(X[bi, wi + i - 1, hi + j - 1, 3]))
    
    def dState(self, Xstate):
        fX = self.filter(Xstate, self.filters)
        fX = tf.reshape(fX, [Xstate.shape[0], Xstate.shape[1], Xstate.shape[2], 3 * self.channel_n])
        bi, i, j, ch = tf.indices([Xstate.shape[0], Xstate.shape[1], Xstate.shape[2], self.channel_n * 4])
        X = tf.select(ch < self.channel_n, Xstate[bi, i, j, ch], fX[bi, i, j, ch - self.channel_n])
        X = GELU(X @ self.fc1 + self.fc1_bias)
        X = X @ self.fc2 + self.fc2_bias
        return tf.reshape(X, [Xstate.shape[0], Xstate.shape[1], Xstate.shape[2], self.channel_n])
    
    def step(self, Xstate):
        dS = self.dState(Xstate)

        #if no active neighbors, dont activate
        activate = tf.float(self.max_neighbor_alpha(Xstate) > 0.1)

        Xshape = Xstate.shape
        mask = self.rand([Xshape[0], Xshape[1], Xshape[2], 1]) < self.fire_rate

        return tf.reshape(tf.clamp((Xstate + tf.select(mask, dS, 0.0)) * activate, -1.0, 1.0), Xstate.shape)
    
    def rand(self, shape):
        self.seed = tf.pcg(self.seed)

        indices = tf.indices(shape)
        element_index = 0
        for i in range(len(shape)):
            element_index = element_index * shape[i] + indices[i]
        return tf.pcgf(tf.uint(element_index) + self.seed)
    
def get_target_batch(image):
    _, wi, hi, ch = tf.indices([BATCH_SIZE, image.shape[0], image.shape[1], image.shape[2]])
    return image[wi, hi, ch]

def rand_range(lo, hi, seed):
    return tf.lerp(lo, hi, tf.pcgf(tf.uint(seed)))

def corruption_mask(shape, seed):
    bi, wi, hi, _ = tf.indices([shape[0], shape[1], shape[2], 1])

    seed = tf.int(tf.pcg(tf.uint(seed + bi)))
    posx = tf.float(shape[1]) * rand_range(0.25, 0.75, seed*3 + 123)
    posy = tf.float(shape[2]) * rand_range(0.25, 0.75, seed*3 + 456)
    rad = rand_range(1.0, rand_range(1.0, 15.0, seed*38 + 51854), seed*3 + 789)

    xi = tf.float(wi)
    yi = tf.float(hi)
    dist = tf.sqrt((xi - posx)**2.0 + (yi - posy)**2.0)
    mask = tf.select(dist >= rad, 1.0, 0.0)
    return mask

def to_rgba(X, ch_offset=0):
    bi, i, j, ch = tf.indices([X.shape[0], X.shape[1], X.shape[2], 4])
    return X[bi, i, j, ch + ch_offset]

def batch_to_img(batch):
    grid_size = tf.int(tf.ceil(tf.sqrt(tf.float(batch.shape[0]))))
    wi, hi, ch = tf.indices([grid_size * batch.shape[1], grid_size * batch.shape[2], 3])
    bi = wi / batch.shape[1] + grid_size * (hi / batch.shape[2])
    color = tf.select(bi < batch.shape[0], batch[bi, wi % batch.shape[1], hi % batch.shape[2], ch], 1.0)
    alpha = tf.select(bi < batch.shape[0], batch[bi, wi % batch.shape[1], hi % batch.shape[2], 3], 1.0)
    res = tf.abs(1.0 - alpha + color)
    return res

class CATrain(tf.Module):
    def __init__(self, train_steps = 25, corrupt_every_n = 2):
        super().__init__()
        self.opt = tf.optimizers.adam(CAModel(), clip = 0.01)
        self.opt.set_clipping_type(tf.clipping.norm)

        self.train_steps = train_steps
        self.corrupt_every_n = corrupt_every_n

        self.train_size = TARGET_SIZE + 2 * TARGET_PADDING
        self.pool = tf.Parameter([POOL_SIZE, self.train_size, self.train_size, self.opt.net.channel_n], tf.float32)

        self.image = tf.Parameter([self.train_size, self.train_size, 4], tf.float32)

    def train_step(self, batch_ids):
        target = get_target_batch(self.image)

        model = self.opt.net
    
        bi = tf.indices([BATCH_SIZE])[0]
        corruption_frame = tf.int(tf.pcg(model.seed[0] + tf.uint(bi + 5451))) % (self.corrupt_every_n * self.train_steps)
        corruption_frame = tf.reshape(corruption_frame, [BATCH_SIZE, 1, 1, 1])

        bi, wi, hi, ch = tf.indices([BATCH_SIZE, self.train_size, self.train_size, model.channel_n])
        state = self.pool[batch_ids[bi], wi, hi, ch]

        Lbatch = tf.mean(tf.mean(tf.mean((to_rgba(state) - target)**2.0)))
        maxL = tf.max(Lbatch)

        #restart the state with the largest loss
        do_restart = Lbatch == maxL
        is_center = (wi == self.train_size//2) & (hi == self.train_size//2) & (ch == 3)
        state = tf.select(do_restart[bi], tf.select(is_center, 1.0, 0.0), state) #initialize with the seed

        mask = corruption_mask(target.shape, tf.int(model.seed[0]))
        corruptor = tf.lerp(0.99, 1.01, model.rand(state.shape)) * mask

        #run the model for a few steps
        for i in range(self.train_steps):
            state = model.step(state)
            state *= tf.select(i == corruption_frame, corruptor, 1.0)

        #loss is the difference between the corrupted and target images
        #reduction over one dimension at a time
        meanL = tf.mean(tf.mean(tf.mean(tf.mean((to_rgba(state) - target)**2.0))))

        self.opt.step(meanL)

        #update the pool with the new state
        self.pool[batch_ids[bi], wi, hi, ch] = state

        return meanL, state

def save_model(model, filename):
    with open(filename, "wb") as f:
        all_params = model.parameters()
        numpy_params = [p.numpy for p in all_params]
        np.savez(f, *numpy_params)

def load_model(model, filename):
    with open(filename, "rb") as f:
        npz = np.load(f)
        arrays = [npz[k] for k in npz.files]
        tensors = [tf.tensor(a) for a in arrays]
        model.update_parameters(tensors)

def inference_step():
    model = CAModel()
    model.initialize_input()

    input_state = tf.input([1, -1, -1, model.channel_n], tf.float32)

    input_params = tf.input([-1], tf.float32)

    mousex = tf.round(input_params[0] * tf.float(input_state.shape[1]))
    mousey = tf.round(input_params[1] * tf.float(input_state.shape[2]))
    press = input_params[2]
    rad = input_params[3]
    ch_offset = tf.int(input_params[4])
    model.fire_rate = input_params[5]
    img_scale = input_params[6]

    ch, wi, hi, ch = input_state.indices
    dist = tf.sqrt((tf.float(wi) - mousex)**2.0 + (tf.float(hi) - mousey)**2.0)
    mask = tf.select((dist < rad) & (press > 0.5), 0.0, 1.0)

    seedstate = tf.select(press < -0.5, tf.select((dist < rad) & (ch == 3), 1.0, 0.0), 0.0)

    input_state = input_state * mask + seedstate

    input_state = model.step(input_state)

    output_image = img_scale*batch_to_img(to_rgba(input_state, ch_offset))

    return output_image, input_state, model.seed

def optimization_step():
    train = CATrain()
    train.initialize_input()

    batch_ids = tf.input([BATCH_SIZE], tf.int32)
    params = tf.input([-1], tf.float32)

    train.opt.learning_rate = params[0]
    ch_offset = tf.int(params[1])
    train.opt.net.fire_rate = params[2]
    img_scale = tf.float(params[3])

    loss, state = train.train_step(batch_ids)

    output_image = img_scale*batch_to_img(to_rgba(state, ch_offset))

    params = train.parameters()
    params.append(loss)
    params.append(output_image)
    return params