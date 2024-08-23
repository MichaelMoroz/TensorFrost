import numpy as np
import TensorFrost as tf
import math

from vec3 import *

def normalize_vec(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z])

def quaternion_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def axis_angle_quaternion(axis, angle):
    axis = normalize_vec(axis)
    angle /= 2
    return np.array([np.cos(angle), axis[0] * np.sin(angle), axis[1] * np.sin(angle), axis[2] * np.sin(angle)])

FIXED_POINT_SIZE = 4096.0

def float2int(f):
    return tf.int(f * FIXED_POINT_SIZE)

def int2float(i):
    return tf.float(i) / FIXED_POINT_SIZE
    
class Camera(tf.Module):
    def __init__(self, position = [0.0, 0.0, 0.0], quaternion = [1.0, 0.0, 0.0, 0.0], W = 512, H = 512, focal_length = 1.0, angular_speed = 0.005, rot_angular_speed = 0.5, camera_speed = 0.01):
        super().__init__()
        self.camera = tf.Parameter([4, 3], tf.float32, requires_grad = False)
        self.params = tf.Parameter([3], tf.float32)
        self.position = np.array(position, dtype=np.float32)
        self.quaternion = np.array(quaternion, dtype=np.float32)
        self.W = W
        self.H = H
        self.min_res = min(W, H)
        self.focal_length = focal_length
        self.angular_speed = angular_speed
        self.camera_speed = camera_speed
        self.rot_angular_speed = rot_angular_speed
        self.pmx = 0
        self.pmy = 0
        self.brightness = 1.0
        self.distance_clip = 10.0
        self.point_radius = 1.0

    #Compiler only
    def assert_parameters(self):
        #initialize camera vectors
        self.cam_p = vec3(self.camera[0, 0], self.camera[0, 1], self.camera[0, 2])
        self.cam_f = vec3(self.camera[3, 0], self.camera[3, 1], self.camera[3, 2])
        self.cam_u = vec3(self.camera[2, 0], self.camera[2, 1], self.camera[2, 2])
        self.cam_v = vec3(self.camera[1, 0], self.camera[1, 1], self.camera[1, 2])
        self.brightness = self.params[0]
        self.distance_clip = self.params[1]
        self.point_radius = self.params[2]

    def uv_to_ij(self, u, v):
        i = v * tf.float(self.min_res) + 0.5 * tf.float(self.H)
        j = u * tf.float(self.min_res) + 0.5 * tf.float(self.W)
        return i, j
    
    def ij_to_uv(self, i, j):
        u = (tf.float(j) - 0.5 * tf.float(self.W)) / tf.float(self.min_res)
        v = (tf.float(i) - 0.5 * tf.float(self.H)) / tf.float(self.min_res)
        return u, v
    
    def get_ray(self, u, v):
        return normalize(self.cam_f * self.focal_length + self.cam_u * u + self.cam_v * v)
    
    def get_ray_dir(self,i, j):
        u, v = self.ij_to_uv(i, j)
        return self.get_ray(u, v)

    def get_rays(self):
        i, j = tf.indices([self.H, self.W])
        u, v = self.ij_to_uv(i, j)
        return self.cam_p, self.get_ray(u, v)
    
    def project(self, p):
        dpos = p - self.cam_p
        norm = self.focal_length / dot(dpos, self.cam_f)
        u = dot(dpos, self.cam_u) * norm
        v = dot(dpos, self.cam_v) * norm
        z = dot(dpos, self.cam_f)
        return u, v, z
    
    def create_image(self):
        image = tf.buffer([self.H, self.W, 3], tf.int32)
        image[image.indices] = 0
        return image
    
    def splat_point_additive(self, image, x, y, z, color, rad_mul = 1.0):
        pos = vec3(x, y, z)
        u, v, z = self.project(pos)
        i, j = self.uv_to_ij(u, v)

        is_inside = (i >= 0.0) & (i < tf.float(self.H)) & (j >= 0.0) & (j < tf.float(self.W)) & (z > 0.0)

        #brightness is proportional to the inverse square of the distance
        brightness = self.brightness * tf.clamp(self.distance_clip / (z * z), 0.0, 1.0)
        render_rad = tf.clamp(self.point_radius * rad_mul, 1.0, 10.0)

        def add(i, j, color, brightness):
            tf.scatterAdd(image[i, j, 0], float2int(brightness*color.x))
            tf.scatterAdd(image[i, j, 1], float2int(brightness*color.y))
            tf.scatterAdd(image[i, j, 2], float2int(brightness*color.z))
        
        with tf.if_cond(is_inside):
            xi = tf.int(i)
            yi = tf.int(j)
            radius = tf.int(tf.ceil(render_rad))
            with tf.loop(-radius, radius + 1) as ii:
                with tf.loop(-radius, radius + 1) as jj:
                    i_new = xi + ii
                    j_new = yi + jj
                    dx = tf.float(i_new) - i
                    dy = tf.float(j_new) - j
                    dist = tf.sqrt(dx*dx + dy*dy)
                    weight = tf.exp(- 2.0*dist*dist / (render_rad * render_rad)) / (math.pi * render_rad * render_rad)
                    add(i_new, j_new, color, brightness * weight)

    #Host only
    def axis(self, axis):
        return quaternion_to_matrix(self.quaternion)[axis, :]
    
    def move_axis(self, axis, distance):
        self.position += normalize_vec(self.axis(axis)) * distance

    def rotate_axis(self, axis, angle):
        self.quaternion = quaternion_multiply(self.quaternion, axis_angle_quaternion(self.axis(axis), angle))
    
    def get_camera_matrix(self):
        return np.stack([self.position, *quaternion_to_matrix(self.quaternion)])
    
    def update_params(self):
        self.camera = tf.tensor(self.get_camera_matrix())
        all_params = [self.brightness, self.distance_clip, self.point_radius]
        self.params = tf.tensor(np.array(all_params, dtype=np.float32))

    def update(self):
        mx, my = tf.get_mouse_position()

        if tf.is_mouse_button_pressed(tf.MOUSE_BUTTON_0):
            self.rotate_axis(0, (mx - self.pmx) * self.angular_speed)
            self.rotate_axis(1, (my - self.pmy) * self.angular_speed)

        if tf.is_key_pressed(tf.KEY_W):
            self.move_axis(2, self.camera_speed)
        if tf.is_key_pressed(tf.KEY_S):
            self.move_axis(2, -self.camera_speed)

        if tf.is_key_pressed(tf.KEY_A):
            self.move_axis(1, -self.camera_speed)
        if tf.is_key_pressed(tf.KEY_D):
            self.move_axis(1, self.camera_speed)

        if tf.is_key_pressed(tf.KEY_Q):
            self.rotate_axis(2, self.angular_speed*2)
        if tf.is_key_pressed(tf.KEY_E):
            self.rotate_axis(2, -self.angular_speed*2)

        self.pmx = mx
        self.pmy = my

        self.update_params()