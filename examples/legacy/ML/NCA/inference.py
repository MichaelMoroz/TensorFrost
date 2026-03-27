import TensorFrost as tf
import numpy as np
import time

from nca import *

tf.initialize(tf.opengl)

inference_step = tf.compile(inference_step)

#initialize the model
model = CAModel()
model.initialize_parameters()

load_model(model, "model.npz")

tf.window.show(INFERENCE_SIZE_X*2, INFERENCE_SIZE_Y*2, "Neural Cellular Automata")

prev_time = time.time()
iterations = 0

input_state_np = np.zeros([1, INFERENCE_SIZE_Y, INFERENCE_SIZE_X, model.channel_n], np.float32)
#pad channels with zeros
input_state_np = np.pad(input_state_np, [(0, 0), (0, 0), (0, 0), (0, model.channel_n - input_state_np.shape[3])], 'constant')
input_state = tf.tensor(input_state_np)

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

    mousex = 1 - my / max(0.1, wy)
    mousey =     mx / max(0.1, wx)
    right_mouse = float(tf.window.is_mouse_button_pressed(tf.window.MOUSE_BUTTON_0))
    left_mouse = -float(tf.window.is_mouse_button_pressed(tf.window.MOUSE_BUTTON_1))
    press = right_mouse + left_mouse
    image, input_state, seed = inference_step(model, input_state, np.array([mousex, mousey, press, mouse_radius, channel_offset, firing_rate, image_scale], np.float32))
    model.seed = seed

    tf.imgui.text("Iterations: %d" % iterations)

    mouse_radius = tf.imgui.slider("Mouse Radius", mouse_radius, 1.0, 10.0)
    channel_offset = tf.imgui.slider("Channel Offset (render)", channel_offset, 0, model.channel_n - 4)
    firing_rate = tf.imgui.slider("Firing Rate", firing_rate, 0.0, 1.0)
    image_scale = tf.imgui.slider("Image Scale", image_scale, 0.01, 2.0)

    if tf.imgui.button("Reset Input State"):
        input_state = tf.tensor(input_state_np)

    tf.window.render_frame(image)

    prev_time = cur_time
    iterations += 1