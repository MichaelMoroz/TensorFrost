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

from nca import *

tf.initialize(tf.opengl)

#https://distill.pub/2020/growing-ca/

TARGET_EMOJI = "😀" #😀🦋🦎🎄🍓

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

#load the target emoji
target = load_emoji(TARGET_EMOJI)

#load from png file
target = PIL.Image.open("D:/garfield.png")
target = target.resize((TARGET_SIZE, TARGET_SIZE), PIL.Image.LANCZOS)
target = np.array(target)
target = np.float32(target)/255.0
target = target[..., :] * target[..., 3:4]

# #plot the target image
# plt.imshow(target)
# plt.axis('off')
# plt.show()

#flip the image on the y axis
target = target[::-1, :]

inference_step = tf.compile(inference_step)
train_step = tf.compile(optimization_step)

#add padding to the target image
target = np.pad(target, [(TARGET_PADDING, TARGET_PADDING), (TARGET_PADDING, TARGET_PADDING), (0, 0)], 'constant')

#initialize the model
trainer = CATrain()
trainer.initialize_parameters()

optimizer = trainer.opt
model = optimizer.net

#initialize the seed
model.seed = tf.tensor(np.array([0], np.uint32))

#initialize the filters
sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
laplace = np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]], np.float32)
filters = np.stack([sobel, sobel.T, laplace], axis = 0)
model.filters = tf.tensor(filters)

#initilize the image
trainer.image = tf.tensor(target)

def initialize_pool():
    #initialize the pool with a white point at the center
    pool = np.zeros([POOL_SIZE, target.shape[0], target.shape[1], model.channel_n], np.float32)
    pool[:, pool.shape[1]//2, pool.shape[2]//2, 3:] = 1.0

    #pad to the channel size
    pool = np.pad(pool, [(0, 0), (0, 0), (0, 0), (0, model.channel_n - pool.shape[3])], 'constant')
    return tf.tensor(pool)

trainer.pool = initialize_pool()

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

input_state_np = np.zeros([1, INFERENCE_SIZE_Y, INFERENCE_SIZE_Y, model.channel_n], np.float32)
input_state_np[0, INFERENCE_SIZE_Y//2, INFERENCE_SIZE_X//2, 3:] = 1.0
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
    mx, my = tf.window.get_mouse_position()
    wx, wy = tf.window.get_size()
    cur_time = time.time()
    delta_time = cur_time - prev_time
    tf.imgui.text("Frame time: %.3f ms" % (delta_time * 1000.0))

    lr = schedule(iterations)

    #if(iterations == 4000):
    #    firing_rate = 1.0

    if train:
        batch_ids = np.random.choice(POOL_SIZE, BATCH_SIZE, replace = False)
        outputs = train_step(trainer, batch_ids, np.array([lr*0.1, channel_offset, firing_rate, image_scale], np.float32))
        trainer.update_parameters(outputs[:-2])
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
        trainer.pool = initialize_pool()

    if tf.imgui.button("Save Model"):
        save_model(model, "model.npz")

    if tf.imgui.button("Load Model"):
        load_model(model, "model.npz")

    tf.window.render_frame(image)

    prev_time = cur_time
    iterations += 1
