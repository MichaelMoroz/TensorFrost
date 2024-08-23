import TensorFrost as tf

from utils import *
from camera import *

def CompileVisualizer(W, H, focal_length):
    def Visualizer():
        camera = Camera(W = W, H = H, focal_length=focal_length)
        camera.initialize_input()

        image = camera.create_image()

        walkers = tf.input([-1, -1, 3], tf.float32)

        point_num = walkers.shape[0] * walkers.shape[1]

        id, = tf.indices([point_num])
        walker_id = id % walkers.shape[0]
        electron_id = id / walkers.shape[0]
       
        x_pos = walkers[walker_id, electron_id, 0]
        y_pos = walkers[walker_id, electron_id, 1]
        z_pos = walkers[walker_id, electron_id, 2]

        camera.splat_point_additive(image, x_pos, y_pos, z_pos, vec3(1.0, 1.0, 1.0))

        fimage = int2float(image)
        return fimage
    
    vis = tf.compile(Visualizer)

    return vis

