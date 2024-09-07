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
TARGET_PADDING = 16   # Number of pixels used to pad the target image border
TARGET_SIZE = 40
BATCH_SIZE = 16
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.9

TARGET_EMOJI = "🦎"

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

class CAModel(tf.Module):
    def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.hidden_size = 32

        self.fc1 = tf.Parameter([channel_n * 3, self.hidden_size], tf.float32)
        self.fc1_bias = tf.Parameter([self.hidden_size], tf.float32, random_scale = 0.0)
        self.fc2 = tf.Parameter([self.hidden_size, channel_n], tf.float32, random_scale = 0.0)
        self.fc2_bias = tf.Parameter([channel_n], tf.float32, random_scale = 0.0)

        self.sobel = tf.Parameter([3, 3], tf.float32, random_scale = 0.0, requires_grad = False)
        self.seed = tf.Parameter([1], tf.uint32, random_scale = 0.0, requires_grad = False)

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

        mask = self.rand(Xstate.shape) < self.fire_rate
        return (Xstate + tf.select(mask, dS, 0.0)) * activate
    
    def iterate(self, Xstate, steps=1):
        for i in range(steps):
            Xstate = self.step(Xstate)
        return Xstate
    
    def rand(self, shape):
        self.seed = tf.pcg(self.seed)

        indices = tf.indices(shape)
        element_index = 0
        for i in range(len(shape)):
            element_index = element_index * shape[i] + indices[i]
        return tf.pcgf(tf.uint(element_index) + self.seed)

lr = 0.0005

def get_model_optimizer():
    model = CAModel()
    opt = tf.optimizers.adam(model, learning_rate = lr, clip = 0.1)
    return model, opt

def rand_range(lo, hi, seed):
    return tf.lerp(lo, hi, tf.pcgf(tf.uint(seed)))

def create_corrupted_batch(image, seed):
    bi, wi, hi, ch = tf.indices([BATCH_SIZE, image.shape[0], image.shape[1], image.shape[2]])

    seed = tf.int(tf.pcg(tf.uint(seed + bi)))
    posx = tf.float(image.shape[0]) * rand_range(0.3, 0.7, seed*3 + 123)
    posy = tf.float(image.shape[1]) * rand_range(0.3, 0.7, seed*3 + 456)
    rad = rand_range(5.0, 10.0, seed*3 + 789)

    xi = tf.float(wi)
    yi = tf.float(hi)
    dist = tf.sqrt((xi - posx)**2.0 + (yi - posy)**2.0)
    mask = tf.select(dist >= rad, 1.0, 0.0)

    return image[wi, hi, ch] * mask, image[wi, hi, ch]

def add_hidden_channels(image, channel_n):
    bi, wi, hi, ch = tf.indices([image.shape[0], image.shape[1], image.shape[2], channel_n])
    return tf.select(ch < image.shape[2], image[bi, wi, hi, ch], 0.0)    

def to_rgba(X):
    bi, i, j, ch = tf.indices([X.shape[0], X.shape[1], X.shape[2], 4])
    return X[bi, i, j, ch]

def batch_to_img(batch):
    grid_size = int(np.ceil(np.sqrt(BATCH_SIZE)))
    wi, hi, ch = tf.indices([grid_size * batch.shape[1], grid_size * batch.shape[2], 3])
    bi = wi / batch.shape[1] + grid_size * (hi / batch.shape[2])
    return tf.select(bi < batch.shape[0], batch[bi, wi % batch.shape[1], hi % batch.shape[2], ch], 1.0)

def optimization_step():
    model, opt = get_model_optimizer()
    opt.initialize_input()

    #target image
    size = TARGET_SIZE + 2 * TARGET_PADDING
    image = tf.input([size, size, 4], tf.float32)

    #create corrupted image set
    corrupted, target = create_corrupted_batch(image, 0)

    #add hidden channels to the corrupted image
    state = add_hidden_channels(corrupted, model.channel_n)

    #run the model for 10 steps
    state = model.iterate(state, 9)

    #extract the output images
    output = to_rgba(state)

    #loss is the difference between the corrupted and target images
    L = tf.mean(tf.mean(tf.mean(tf.mean((output - target)**2.0))))

    opt.step(L)

    output_image = batch_to_img(output)

    params = opt.parameters()
    params.append(L)
    params.append(output_image)
    return params

train_step = tf.compile(optimization_step)

#load the target emoji
target = load_emoji(TARGET_EMOJI)

#add padding to the target image
target = np.pad(target, [(TARGET_PADDING, TARGET_PADDING), (TARGET_PADDING, TARGET_PADDING), (0, 0)], 'constant')

#plot the target emoji
print("Target shape:", target.shape)
plt.imshow(target[:, :, 3])
plt.show()

#initialize the model
model, opt = get_model_optimizer()
opt.initialize_parameters()

#initialize the seed
model.seed = tf.tensor(np.array([0], np.uint32))

#initialize the sobel filter
sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
model.sobel = tf.tensor(sobel)

#target image
target_tf = tf.tensor(target)

# #run a step of the optimization
# outputs = train_step(opt, target_tf)
# opt.update_parameters(outputs[:-2])
# L = outputs[-2].numpy
# output = outputs[-1].numpy

# print("Loss:", L)

#run the optimization for 1000 steps
# progress_bar = tqdm(range(10))
# for i in progress_bar:
#     outputs = train_step(opt, target_tf)
#     opt.update_parameters(outputs[:-2])
#     L = outputs[-2].numpy
#     progress_bar.set_postfix(loss = L)

# #plot the output images
# output = outputs[-1].numpy
# plt.imshow(output[:, :, 3])
# plt.show()

ImageSize = 1080

tf.window.show(ImageSize, ImageSize, "Neural Cellular Automata")

prev_time = time.time()

while not tf.window.should_close():
    cur_time = time.time() 
    time_tf = tf.tensor(np.array([cur_time], np.float32))
    mx, my = tf.window.get_mouse_position()
    cur_time = time.time()
    delta_time = cur_time - prev_time
    tf.imgui.text("Frame time: %.3f ms" % (delta_time * 1000.0))

    outputs = train_step(opt, target_tf)
    opt.update_parameters(outputs[:-2])
    L = outputs[-2].numpy
    tf.imgui.text("Loss: %.3f" % L)

    tf.window.render_frame(outputs[-1])

    prev_time = cur_time