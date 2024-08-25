import TensorFrost as tf

from utils import *
from camera import *

def CompileVisualizer(W, H, focal_length):
    def Visualizer():
        camera = Camera(W = W, H = H, focal_length=focal_length)
        camera.initialize_input()

        image = camera.create_image()

        walkers = tf.input([-1, -1, 3], tf.float32)
        atoms = tf.input([-1, 4], tf.float32)

        point_num = walkers.shape[0] * walkers.shape[1]

        id, = tf.indices([point_num])
        walker_id = id % walkers.shape[0]
        electron_id = id / walkers.shape[0]
       
        x_pos = walkers[walker_id, electron_id, 0]
        y_pos = walkers[walker_id, electron_id, 1]
        z_pos = walkers[walker_id, electron_id, 2]

        camera.splat_point_additive(image, x_pos, y_pos, z_pos, vec3(1.0, 1.0, 1.0))

        
        atom_num = atoms.shape[0]
        atom_id, = tf.indices([atom_num])
        x_pos = atoms[atom_id, 0]
        y_pos = atoms[atom_id, 1]
        z_pos = atoms[atom_id, 2]

        camera.splat_point_additive(image, x_pos, y_pos, z_pos, 1e5*vec3(0.2, 0.2, 1.0), rad_mul = 3.0)


        fimage = 1.0 - tf.tanh(tf.pow(int2float(image), 0.25))
        return fimage
    
    vis = tf.compile(Visualizer)

    return vis

