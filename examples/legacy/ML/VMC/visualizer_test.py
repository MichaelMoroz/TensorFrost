import TensorFrost as tf
from TensorFrost import imgui
from TensorFrost import window
import numpy as np
import matplotlib.pyplot as plt

from visualizer import *

tf.initialize(tf.opengl)

walkers = np.random.randn(65536, 8, 3).astype(np.float32)
walkers_tf = tf.tensor(walkers)

atoms = np.random.randn(16, 4).astype(np.float32)
atoms_tf = tf.tensor(atoms)

vis = CompileVisualizer(1920, 1080, 1.0)

cam = Camera(position=np.array([0.0, 0.0, 0.0]), quaternion=np.array([0.0, 0.0, 0.0, 1.0]), W=1920, H=1080, focal_length=1.0, angular_speed = 0.005, camera_speed = 0.01)
cam.initialize_parameters()

window.show(cam.W, cam.H, "Walker renderer")

random = np.random.randn(128).astype(np.float32)

while not window.should_close():
    cam.update()

    imgui.text("Error rel:, {:.3f} %".format(100.0 * np.abs((12.0) / 30.0)))
    cam.angular_speed = imgui.slider("Angular speed", cam.angular_speed, 0.0, 0.01)
    cam.camera_speed = imgui.slider("Camera speed", cam.camera_speed, 0.0, 0.5)
    cam.focal_length = imgui.slider("Focal length", cam.focal_length, 0.1, 10.0)
    cam.brightness = imgui.slider("Brightness", cam.brightness, 0.0, 5.0)
    cam.distance_clip = imgui.slider("Distance clip", cam.distance_clip, 0.0, 100.0)
    cam.point_radius = imgui.slider("Point radius", cam.point_radius, 0.0, 10.0)

    imgui.plotlines("Random", random, overlay_text="Random plot", scale_min=-10.0, scale_max=10.0, graph_size=(0, 200))

    #imgui.add_background_text("Hello world", (100, 100), (255, 255, 255, 255))

    atom_screen = ProjectPoints(cam, atoms)
    for i in range(atoms.shape[0]):
        if atom_screen[i, 2] > 0.0:
            imgui.add_background_text("Atom", (atom_screen[i, 1], atom_screen[i, 0]), (255, 255, 255, 255))

    cam.update_tensors()
    window.render_frame(vis(cam, walkers_tf, atoms_tf))
    

    
  
    