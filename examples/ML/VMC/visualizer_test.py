import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt

from visualizer import *

tf.initialize(tf.opengl)

walkers = np.random.randn(16384, 16, 3).astype(np.float32)
walkers_tf = tf.tensor(walkers)

atoms = np.random.randn(16, 4).astype(np.float32)
atoms_tf = tf.tensor(atoms)

vis = CompileVisualizer(1280, 720, 1.0)

cam = Camera(position=np.array([0.0, 0.0, 0.0]), quaternion=np.array([0.0, 0.0, 0.0, 1.0]), W=1280, H=720, focal_length=1.0, angular_speed = 0.005, camera_speed = 0.01)
cam.initialize_parameters()

tf.show_window(cam.W, cam.H, "Walker renderer")

while not tf.window_should_close():
    tf.imgui_text("Error rel:, {:.3f} %".format(100.0 * np.abs((12.0) / 30.0)))
    cam.angular_speed = tf.imgui_slider("Angular speed", cam.angular_speed, 0.0, 0.01)
    cam.camera_speed = tf.imgui_slider("Camera speed", cam.camera_speed, 0.0, 0.5)
    cam.focal_length = tf.imgui_slider("Focal length", cam.focal_length, 0.1, 10.0)
    cam.brightness = tf.imgui_slider("Brightness", cam.brightness, 0.0, 5.0)
    cam.distance_clip = tf.imgui_slider("Distance clip", cam.distance_clip, 0.0, 100.0)
    cam.point_radius = tf.imgui_slider("Point radius", cam.point_radius, 0.0, 10.0)

    cam.update()
    tf.render_frame(vis(cam, walkers_tf, atoms_tf))
    

    
  
    