import TensorFrost as tf
import numpy as np
import time
from tqdm import tqdm
import os
import io
import PIL.Image, PIL.ImageDraw
import base64
import zipfile
import json
import requests
import matplotlib.pyplot as plt

tf.initialize(tf.opengl)

#https://distill.pub/2020/growing-ca/

CHANNEL_N = 16        # Number of CA state channels
TARGET_PADDING = 8   # Number of pixels used to pad the target image border
TARGET_SIZE = 40
BATCH_SIZE = 3*3
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.75

INFERENCE_SIZE = 384

TARGET_EMOJI = "🦎" #😀🦋🦎🎄🍓

def load_image(url, max_size=TARGET_SIZE):
    r = requests.get(url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), PIL.Image.LANCZOS)
    img = np.float32(img)/255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img

def load_emoji(emoji):
    code = hex(ord(emoji))[2:].lower()
    url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
    return load_image(url)

def GELU(X):
    return 0.5*X*(1.0 + tf.tanh(np.sqrt(2.0/np.pi) * (X + 0.044715 * (X * X * X))))

def LeakyReLU(X):
    return tf.select(X > 0.0, X, 0.01*X)

#load the target emoji
#target = load_emoji(TARGET_EMOJI)

#load from png file
target = PIL.Image.open("examples/Rendering/test.png")
target.thumbnail((TARGET_SIZE, TARGET_SIZE), PIL.Image.LANCZOS)
target = np.array(target)
target = np.float32(target)/255.0
target = target[..., :] * target[..., 3:4]

# #plot the target image
# plt.imshow(target)
# plt.axis('off')
# plt.show()

#flip the image on the y axis
target = target[::-1, :]


class CAModel(tf.Module):
    def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.hidden_size = 128

        train_size = TARGET_SIZE + 2 * TARGET_PADDING
        self.pool = tf.Parameter([POOL_SIZE, train_size, train_size, channel_n], tf.float32, optimize = False)

        self.fc1 = tf.Parameter([channel_n * 3, self.hidden_size], tf.float32)
        self.fc1_bias = tf.Parameter([self.hidden_size], tf.float32, random_scale = 0.0)
        self.fc2 = tf.Parameter([self.hidden_size, channel_n], tf.float32, random_scale = 0.0)
        self.fc2_bias = tf.Parameter([channel_n], tf.float32, random_scale = 0.0)

        self.sobel = tf.Parameter([3, 3], tf.float32, random_scale = 0.0, optimize = False)
        self.seed = tf.Parameter([1], tf.uint32, random_scale = 0.0, optimize = False)

    def assert_parameters(self):
        self.fc2 = tf.assert_tensor(self.fc2, [self.fc1.shape[1], self.fc2.shape[1]], tf.float32)

    def filter(self, X, W):
        bi, wi, hi, ch, it = tf.indices([X.shape[0], X.shape[1], X.shape[2], X.shape[3], 9])
        i, j = it % 3, it / 3
        conv = tf.sum(X[bi, wi + i - 1, hi + j - 1, ch] * W[i, j])
        return conv
    
    def max_neighbor_alpha(self, X):
        bi, wi, hi, it = tf.indices([X.shape[0], X.shape[1], X.shape[2], 9])
        i, j = it % 3, it / 3
        return tf.unsqueeze(tf.max(X[bi, wi + i - 1, hi + j - 1, 3]))
    
    def dState(self, Xstate):
        dX = self.filter(Xstate, self.sobel)
        dY = self.filter(Xstate, self.sobel.T)
        bi, i, j, ch = tf.indices([Xstate.shape[0], Xstate.shape[1], Xstate.shape[2], self.channel_n * 3])
        X = tf.select(ch < self.channel_n, Xstate[bi, i, j, ch], tf.select(ch < 2 * self.channel_n, dX[bi, i, j, ch - self.channel_n], dY[bi, i, j, ch - 2 * self.channel_n]))
        X = GELU(X @ self.fc1 + self.fc1_bias)
        X = X @ self.fc2 + self.fc2_bias
        return tf.reshape(X, [Xstate.shape[0], Xstate.shape[1], Xstate.shape[2], self.channel_n])
    
    def step(self, Xstate):
        dS = self.dState(Xstate)

        #if no active neighbors, dont activate
        activate = tf.float(self.max_neighbor_alpha(Xstate) > 0.1)

        Xshape = Xstate.shape
        #mask = self.rand(Xshape) < self.fire_rate
        mask = self.rand([Xshape[0], Xshape[1], Xshape[2], 1]) < self.fire_rate
        #noise = self.rand(Xshape) * 2.0 - 1.0

        #boundary conditions
        bi, wi, hi, ch = tf.indices([Xshape[0], Xshape[1], Xshape[2], 1])
        bc = 1.0#tf.select((wi <= 1) | (wi >= Xshape[1] - 2) | (hi <= 1) | (hi >= Xshape[2] - 2), 0.0, 1.0)

        return tf.reshape((Xstate + tf.select(mask, dS, 0.0)) * bc * activate, Xstate.shape)
    
    def rand(self, shape):
        self.seed = tf.pcg(self.seed)

        indices = tf.indices(shape)
        element_index = 0
        for i in range(len(shape)):
            element_index = element_index * shape[i] + indices[i]
        return tf.pcgf(tf.uint(element_index) + self.seed)

def get_model_optimizer():
    model = CAModel()
    opt = tf.optimizers.adam(model, clip = 0.01)
    opt.set_clipping_type(tf.clipping.norm)
    return model, opt

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

def get_target_batch(image):
    _, wi, hi, ch = tf.indices([BATCH_SIZE, image.shape[0], image.shape[1], image.shape[2]])
    return image[wi, hi, ch]

def add_hidden_channels(image, channel_n):
    bi, wi, hi, ch = tf.indices([image.shape[0], image.shape[1], image.shape[2], channel_n])
    return tf.select(ch < image.shape[2], image[bi, wi, hi, ch], 0.0)    

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
    #max_color = tf.max(tf.max(tf.max(res)))
    return res

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

    #run the model for a few steps
    input_state = model.step(input_state)

    #extract the output images
    output_image = img_scale*batch_to_img(to_rgba(input_state, ch_offset))

    return output_image, input_state, model.seed

inference_step = tf.compile(inference_step)

def optimization_step():
    model, opt = get_model_optimizer()
    opt.initialize_input()

    #target image
    size = TARGET_SIZE + 2 * TARGET_PADDING
    image = tf.input([size, size, 4], tf.float32)
    batch_ids = tf.input([BATCH_SIZE], tf.int32)

    params = tf.input([-1], tf.float32)

    max_steps = tf.int(params[0])
    opt.learning_rate = params[1]
    ch_offset = tf.int(params[2])
    model.fire_rate = params[3]
    img_scale = tf.float(params[4])

    #create corrupted image set
    target = get_target_batch(image)
    mask = corruption_mask(target.shape, tf.int(model.seed[0]))

    iters_per_optimizer_step = 25
    corrupt_every_n = 2
    
    bi = tf.indices([BATCH_SIZE])[0]
    corruption_frame = tf.int(tf.pcg(model.seed[0] + tf.uint(bi + 5451))) % (corrupt_every_n * iters_per_optimizer_step)
    corruption_frame = tf.reshape(corruption_frame, [BATCH_SIZE, 1, 1, 1])

    bi, wi, hi, ch = tf.indices([BATCH_SIZE, size, size, model.channel_n])
    state = model.pool[batch_ids[bi], wi, hi, ch]

    Lbatch = tf.mean(tf.mean(tf.mean((to_rgba(state) - target)**2.0)))
    maxL = tf.max(Lbatch)

    #restart the state with the largest loss
    do_restart = Lbatch == maxL
    is_center = (wi == size//2) & (hi == size//2) & (ch == 3)
    state = tf.select(do_restart[bi], tf.select(is_center, 1.0, 0.0), state) #initialize with the seed

    corruptor = tf.lerp(0.99, 1.01, model.rand(state.shape)) * mask

    #run the model for a few steps
    for i in range(iters_per_optimizer_step):
        state = model.step(state)
        state *= tf.select(i == corruption_frame, corruptor, 1.0)

    #loss is the difference between the corrupted and target images
    #reduction over one dimension at a time
    meanL = tf.mean(tf.mean(tf.mean(tf.mean((to_rgba(state) - target)**2.0))))

    opt.step(meanL)
    
    output_image = img_scale*batch_to_img(to_rgba(state, ch_offset))

    #update the pool with the new state
    model.pool[batch_ids[bi], wi, hi, ch] = state

    params = opt.parameters()
    params.append(meanL)
    params.append(output_image)
    return params

train_step = tf.compile(optimization_step)


#add padding to the target image
target = np.pad(target, [(TARGET_PADDING, TARGET_PADDING), (TARGET_PADDING, TARGET_PADDING), (0, 0)], 'constant')

#initialize the model
model, opt = get_model_optimizer()
opt.initialize_parameters()

#initialize the seed
model.seed = tf.tensor(np.array([0], np.uint32))

#initialize the sobel filter
sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
model.sobel = tf.tensor(sobel)

def initialize_pool():
    #initialize the pool with the target image
    #pool = np.repeat(target[np.newaxis, ...], POOL_SIZE, axis = 0)

    #initialize the pool with a white point at the center
    pool = np.zeros([POOL_SIZE, target.shape[0], target.shape[1], model.channel_n], np.float32)
    pool[:, pool.shape[1]//2, pool.shape[2]//2, 3:] = 1.0

    #pad to the channel size
    pool = np.pad(pool, [(0, 0), (0, 0), (0, 0), (0, model.channel_n - pool.shape[3])], 'constant')
    return tf.tensor(pool)

model.pool = initialize_pool()

#target image
target_tf = tf.tensor(target)

ImageSize = 1024

tf.window.show(ImageSize, ImageSize, "Neural Cellular Automata")

prev_time = time.time()

smoothed_loss = 0.0
lrs = [0.05, 0.02, 0.01, 0.002]
steps = [0, 1000, 2000, 3000]

def schedule(iterations):
    #piecewise linear learning rate schedule

    #find the current segment
    for i in range(len(steps) - 1):
        if iterations >= steps[i] and iterations < steps[i + 1]:
            t = (iterations - steps[i]) / (steps[i + 1] - steps[i])
            return lrs[i] * (1.0 - t) + lrs[i + 1] * t
    
    return lrs[-1]

iterations = 0

#input_state_np = target.reshape([1, target.shape[0], target.shape[1], target.shape[2]])
#input_state_np = input_state_np.astype(np.float32)
input_state_np = np.zeros([1, INFERENCE_SIZE, INFERENCE_SIZE, model.channel_n], np.float32)
input_state_np[0, INFERENCE_SIZE//2, INFERENCE_SIZE//2, 3:] = 1.0
#pad channels with zeros
input_state_np = np.pad(input_state_np, [(0, 0), (0, 0), (0, 0), (0, model.channel_n - input_state_np.shape[3])], 'constant')
input_state = tf.tensor(input_state_np)

train = True
mouse_radius = 3.0
channel_offset = 0
firing_rate = CELL_FIRE_RATE
image_scale = 1.0

while not tf.window.should_close():
    cur_time = time.time() 
    time_tf = tf.tensor(np.array([cur_time], np.float32))
    mx, my = tf.window.get_mouse_position()
    wx, wy = tf.window.get_size()
    cur_time = time.time()
    delta_time = cur_time - prev_time
    tf.imgui.text("Frame time: %.3f ms" % (delta_time * 1000.0))

    lr = schedule(iterations)

    if(iterations == 4000):
        firing_rate = 1.0

    if train:
        batch_ids = np.random.choice(POOL_SIZE, BATCH_SIZE, replace = False)
        outputs = train_step(opt, target_tf, batch_ids, np.array([0, lr*0.1, channel_offset, firing_rate, image_scale], np.float32))
        opt.update_parameters(outputs[:-2])
        L = outputs[-2].numpy[0]
        image = outputs[-1]
        smoothed_loss = 0.96 * smoothed_loss + 0.04 * L
    else:
        mousex = 1 - my / max(0.1, wy)
        mousey =     mx / max(0.1, wx)
        right_mouse = float(tf.window.is_mouse_button_pressed(tf.window.MOUSE_BUTTON_0))
        left_mouse = -float(tf.window.is_mouse_button_pressed(tf.window.MOUSE_BUTTON_1))
        press = right_mouse + left_mouse
        image, input_state, seed = inference_step(model, input_state, np.array([mousex, mousey, press, mouse_radius, channel_offset, firing_rate, image_scale], np.float32))
        model.seed = seed

    tf.imgui.text("Loss: %.5f" % smoothed_loss)
    tf.imgui.text("Iterations: %d" % iterations)
    # lr = tf.imgui.slider("Learning Rate", lr, 0.0005, 0.1)
    tf.imgui.text("Learning Rate: %.5f" % lr)


    mouse_radius = tf.imgui.slider("Mouse Radius", mouse_radius, 1.0, 10.0)
    channel_offset = tf.imgui.slider("Channel Offset (render)", channel_offset, 0, model.channel_n - 4)
    firing_rate = tf.imgui.slider("Firing Rate", firing_rate, 0.0, 1.0)
    image_scale = tf.imgui.slider("Image Scale", image_scale, 0.01, 2.0)
    train = tf.imgui.checkbox("Training", train)

    if tf.imgui.button("Reset Input State"):
        input_state = tf.tensor(input_state_np)

    if tf.imgui.button("Reset Pool"):
        model.pool = initialize_pool()

    tf.window.render_frame(image)

    prev_time = cur_time
    iterations += 1