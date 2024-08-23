import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt

from visualizer import *

tf.initialize(tf.opengl)

walkers = np.random.randn(100, 16, 3).astype(np.float32)
walkers_tf = tf.tensor(walkers)

vis = CompileVisualizer(1280, 720, 1.0)

cam = Camera(position=np.array([0.0, 0.0, 0.0]), quaternion=np.array([0.0, 0.0, 0.0, 1.0]), W=1280, H=720, focal_length=1.0, angular_speed = 0.005, camera_speed = 0.01)
cam.initialize_parameters()

tf.show_window(cam.W, cam.H, "Walker renderer")

while not tf.window_should_close():
    cam.controller_update()
    image = vis(cam, walkers_tf)
    tf.render_frame(image)

    
  
    