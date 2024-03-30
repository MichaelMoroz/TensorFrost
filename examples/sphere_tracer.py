import TensorFrost as tf
import numpy as np
import time

tf.initialize(tf.opengl)

S = 1024
eps = 0.001
m_pow = 8.0
max_depth = 50.0
min_angle = 0.0005

class vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def vec3(x, y, z):
        return vec3(x, y, z)
    
    def zero(shape):
        return vec3(tf.zeros(shape, tf.float32), tf.zeros(shape, tf.float32), tf.zeros(shape, tf.float32))
    
    def zero_like(val):
        return vec3.zero(val.x.shape)
    
    def const(val, shape):
        return vec3(tf.const(val, shape, tf.float32), tf.const(val, shape, tf.float32), tf.const(val, shape, tf.float32))
    
    def copy(val):
        vec = vec3.zero(val.x.shape)
        vec.set(val)
        return vec
    
    def set(self, other):
        self.x.val = other.x
        self.y.val = other.y
        self.z.val = other.z
    
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __radd__(self, other):
        return vec3(other.x + self.x, other.y + self.y, other.z + self.z)
    
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __rsub__(self, other):
        return vec3(other.x - self.x, other.y - self.y, other.z - self.z)
    
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    
    def __rmul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    
    def __truediv__(self, other):
        return vec3(self.x / other, self.y / other, self.z / other)
    
    def __neg__(self):
        return vec3(-self.x, -self.y, -self.z)
    
    def __abs__(self):
        return vec3(tf.abs(self.x), tf.abs(self.y), tf.abs(self.z))
    
    def __pow__(self, other):
        return vec3(self.x ** other, self.y ** other, self.z ** other)
    
    def __rpow__(self, other):
        return vec3(other ** self.x, other ** self.y, other ** self.z)
    
def dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z

def cross(a, b):
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)

def length(a):
    return tf.sqrt(dot(a, a))

def normalize(a):
    return a / length(a)

def min(a, b):
    return vec3(tf.min(a.x, b.x), tf.min(a.y, b.y), tf.min(a.z, b.z))

def max(a, b):
    return vec3(tf.max(a.x, b.x), tf.max(a.y, b.y), tf.max(a.z, b.z))

def clamp(a, low, high):
    return vec3(tf.clamp(a.x, low, high), tf.clamp(a.y, low, high), tf.clamp(a.z, low, high))

def mandelbulb(p):
    w = vec3.copy(p)
    m = dot(w,w)
    dz = tf.const(1.0, p.x.shape)
    col = vec3.copy(abs(w))
    def loop_body(i):
        dz.val = m_pow * m ** (0.5*(m_pow - 1.0)) * dz + 1.0
        r = length(w)
        b = m_pow * tf.acos(w.y/r)
        a = m_pow * tf.atan2(w.x, w.z)
        c = r ** m_pow
        w.set(vec3(p.x + c * tf.sin(b) * tf.sin(a), p.y + c * tf.cos(b), p.z + c * tf.sin(b) * tf.cos(a)))
        col.set(min(col, abs(w)))
        m.val = dot(w,w)
        tf.if_cond(m > 256.0, lambda: tf.break_loop())

    tf.loop(loop_body, 0, 3, 1)
    sdf = 0.25 * tf.log(m) * tf.sqrt(m) / dz
    return sdf, col

def calcNormal(p):
    sdf = mandelbulb(p)[0]
    sdfx = mandelbulb(p + vec3(eps, 0.0, 0.0))[0]
    sdfy = mandelbulb(p + vec3(0.0, eps, 0.0))[0]
    sdfz = mandelbulb(p + vec3(0.0, 0.0, eps))[0]
    return normalize(vec3(sdfx - sdf, sdfy - sdf, sdfz - sdf))

def MarchRay(ro, rd, steps=128):
    td = tf.zeros([])
    def loop_body(k):
        sdf = mandelbulb(ro + rd * td)[0]
        td.val += sdf
        tf.if_cond((sdf < min_angle * td) | (td > max_depth), lambda: tf.break_loop())

    tf.loop(loop_body, 0, steps, 1)
    return td

def MarchSoftShadow(ro, rd, w, steps=256):
    td = tf.zeros([])
    psdf = tf.const(1e10, [])
    res = tf.const(1.0, [])
    def loop_body(k):
        sdf = mandelbulb(ro + rd * td)[0]

        y = sdf * sdf / (2.0 * psdf)
        d = tf.sqrt(sdf * sdf - y * y)
        res.val = tf.min(res, d / (w * tf.max(0.0, td - y)))
        psdf.val = sdf

        td.val += sdf
        tf.if_cond((sdf < min_angle * td) | (td > max_depth), lambda: tf.break_loop())

    tf.loop(loop_body, 0, steps, 1)

    res = tf.clamp(res, 0.0, 1.0)
    return res * res * (3.0 - 2.0 * res)

light_dir = vec3(0.577, 0.577, 0.577)

def get_ray(u, v, camera, shape):
    pos = vec3.zero(shape)
    dir = vec3.zero(shape)
    pos.set(vec3(camera[0, 0], camera[0, 1], camera[0, 2]))
    cam_f = vec3(camera[3, 0], camera[3, 1], camera[3, 2])
    cam_u = vec3(camera[2, 0], camera[2, 1], camera[2, 2])
    cam_v = vec3(camera[1, 0], camera[1, 1], camera[1, 2])
    dir.set(cam_f + cam_u * u + cam_v * v)
    return pos, normalize(dir)

def ray_marcher():
    # Camera parameters (pos, cam_axis_x, cam_axis_y, cam_axis_z)
    camera = tf.input([4,3], tf.float32)
    
    N, M = S, S
    canvas = tf.zeros([N, M, 3], tf.float32)
    i, j = tf.indices([N, M])
    min_res = tf.min(N, M)
    v, u = tf.float(i), tf.float(j)
    v = (v - 0.5 * tf.float(N)) / tf.float(min_res)
    u = (u - 0.5 * tf.float(M)) / tf.float(min_res)

    ro, rd = get_ray(u, v, camera, i.shape)
    
    td = MarchRay(ro, rd)

    def if_body():
        hit = ro + rd * td

        # Lighting
        norm = calcNormal(hit)
        #sdf, col = mandelbulb(hit)
        #col1 = clamp(col, 0.6, 1.0)
        
        #shooting shadow ray
        hit += norm * min_angle * td
        shadow = MarchSoftShadow(hit, light_dir, 0.06) + 0.1
        b = shadow * (dot(norm, light_dir) * 0.5 + 0.5)
        canvas[i, j, 0] = b
        canvas[i, j, 1] = b
        canvas[i, j, 2] = b

    tf.if_cond(td < max_depth, if_body)
    
    return [canvas]

raymarch = tf.compile(ray_marcher)

#print(raymarch.list_operations())

def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z])

def quaternion_to_matrix(q):
    """Convert a quaternion into a rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

class Camera:
    def __init__(self, position, quaternion):
        self.position = np.array(position, dtype=np.float32)
        self.quaternion = np.array(quaternion, dtype=np.float32)

    def move_forward(self, distance):
        """Move the camera forward."""
        forward = quaternion_to_matrix(self.quaternion)[2, :]  # z axis
        self.position += normalize(forward) * distance

    def move_right(self, distance):
        """Move the camera right."""
        right = quaternion_to_matrix(self.quaternion)[1, :]  # y axis
        self.position += normalize(right) * distance
    
    def move_up(self, distance):
        """Move the camera up."""
        up = quaternion_to_matrix(self.quaternion)[0, :]
        self.position += normalize(up) * distance

    def rotate_left(self, angle):
        """Rotate the camera around its y axis."""
        right = quaternion_to_matrix(self.quaternion)[1, :]
        q = np.array([np.cos(angle/2), right[0]*np.sin(angle/2), right[1]*np.sin(angle/2), right[2]*np.sin(angle/2)])
        self.quaternion = quaternion_multiply(self.quaternion, q)

    def rotate_up(self, angle):
        """Rotate the camera around its x axis."""
        up = quaternion_to_matrix(self.quaternion)[0, :]
        q = np.array([np.cos(angle/2), up[0]*np.sin(angle/2), up[1]*np.sin(angle/2), up[2]*np.sin(angle/2)])
        self.quaternion = quaternion_multiply(self.quaternion, q)

    def rotate_roll(self, angle):
        """Rotate the camera around its z axis."""
        forward = quaternion_to_matrix(self.quaternion)[2, :]
        q = np.array([np.cos(angle/2), forward[0]*np.sin(angle/2), forward[1]*np.sin(angle/2), forward[2]*np.sin(angle/2)])
        self.quaternion = quaternion_multiply(self.quaternion, q)

    def get_camera_axis_matrix(self):
        """Get the camera axis matrix."""
        return quaternion_to_matrix(self.quaternion)
    
    def get_camera_matrix(self):
        """Get the camera matrix."""
        return np.stack([self.position, *self.get_camera_axis_matrix()])


def render_mandelbulb(camera):
    camera_matrix_tf = tf.tensor(camera.get_camera_matrix())

    img, = raymarch(camera_matrix_tf)

    return img

tf.show_window(S, S, "Sphere tracer")

camera = Camera([0, 0, -2], [1, 0, 0, 0])
pmx, pmy = tf.get_mouse_position()

angular_speed = 0.005
camera_speed = 0.005

while not tf.window_should_close():
    mx, my = tf.get_mouse_position()

    if tf.is_mouse_button_pressed(tf.MOUSE_BUTTON_0):
        camera.rotate_up((mx - pmx) * angular_speed)
        camera.rotate_left((my - pmy) * angular_speed)

    if tf.is_key_pressed(tf.KEY_W):
        camera.move_forward(camera_speed)
    if tf.is_key_pressed(tf.KEY_S):
        camera.move_forward(-camera_speed)

    if tf.is_key_pressed(tf.KEY_A):
        camera.move_right(-camera_speed)
    if tf.is_key_pressed(tf.KEY_D):
        camera.move_right(camera_speed)

    if tf.is_key_pressed(tf.KEY_Q):
        camera.rotate_roll(-angular_speed*2)
    if tf.is_key_pressed(tf.KEY_E):
        camera.rotate_roll(angular_speed*2)

    img = render_mandelbulb(camera)
    tf.render_frame(img)
    
    pmx, pmy = mx, my
    
