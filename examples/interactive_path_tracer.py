import TensorFrost as tf
import numpy as np
import time
import os
import imageio
current_dir = os.path.dirname(os.path.abspath(__file__))

tf.initialize(tf.opengl)

W = 1920
H = 1080
eps = 0.0001
m_pow = 8.0
max_depth = 150.0
min_angle = 0.0005

class vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def zero(shape):
        return vec3(tf.zeros(shape, tf.float32), tf.zeros(shape, tf.float32), tf.zeros(shape, tf.float32))
    
    def zero_like(val):
        return vec3.zero(val.x.shape)
    
    def const(val, shape):
        return vec3(tf.const(val, shape), tf.const(val, shape), tf.const(val, shape))
    
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
    
    def __div__(self, other):
        return vec3(self.x / other, self.y / other, self.z / other)
    
    def __rdiv__(self, other):
        return vec3(other / self.x, other / self.y, other / self.z)
    
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

def mul(a, b):
    return vec3(a.x * b.x, a.y * b.y, a.z * b.z)

def dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z

def cross(a, b):
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)

def length(a):
    return tf.sqrt(dot(a, a))

def distance(a, b):
    return length(a - b)

def normalize(a):
    return a / (length(a) + 1e-5)

def min(a, b):
    return vec3(tf.min(a.x, b.x), tf.min(a.y, b.y), tf.min(a.z, b.z))

def max(a, b):
    return vec3(tf.max(a.x, b.x), tf.max(a.y, b.y), tf.max(a.z, b.z))

def clamp(a, low, high):
    return vec3(tf.clamp(a.x, low, high), tf.clamp(a.y, low, high), tf.clamp(a.z, low, high))

def exp(a):
    return vec3(tf.exp(a.x), tf.exp(a.y), tf.exp(a.z))

def lerp(a, b, t):
    return a + (b - a) * t

def abs(a):
    return vec3(tf.abs(a.x), tf.abs(a.y), tf.abs(a.z))

def reflect(i, n):
    return i - n * 2.0 * dot(n, i)

def sdBox(p, b):
    d = abs(p) - b
    return tf.min(tf.max(d.x, tf.max(d.y, d.z)), 0.0) + length(max(d, vec3(0.0, 0.0, 0.0)))


class Level:
    def __init__(self, scale, ang1, ang2, shift, col):
        self.scale = scale
        self.ang1 = ang1
        self.ang2 = ang2
        self.shift = shift
        self.col = col
    
levels = [
    Level(1.8, -0.12, 0.5, vec3(-2.12, -2.75, 0.49), vec3(0.42, 0.38, 0.19)),
    Level(1.9073, -9.83, -1.16, vec3(-3.508, -3.593, 3.295), vec3(-0.34, 0.12, -0.08)),
    Level(2.02, -1.57, 1.62, vec3(-3.31, 6.19, 1.53), vec3(0.12, -0.09, -0.09)),
    Level(1.66, 1.52, 0.19, vec3(-3.83, -1.94, -1.09), vec3(0.42, 0.38, 0.19)),
    Level(1.58, -1.45, 3.95, vec3(-1.55, -0.13, -2.52), vec3(-1.17, -0.4, -1.0)),
    Level(1.81, -4.84, -2.99, vec3(-2.905, 0.765, -4.165), vec3(0.251, 0.337, 0.161)),
    Level(1.88, 1.52, 4.91, vec3(-4.54, -1.26, 0.1), vec3(-1.0, 0.3, -0.43)),
    Level(2.08, -4.79, 3.16, vec3(-7.43, 5.96, -6.23), vec3(0.16, 0.38, 0.15)),
    Level(2.0773, -9.66, -1.34, vec3(-1.238, -1.533, 1.085), vec3(0.42, 0.38, 0.19)),
    Level(2.0773, -9.66, -1.34, vec3(-1.238, -1.533, 1.085), vec3(0.42, 0.38, 0.19)),
    Level(1.4731, 0.0, 0.0, vec3(-10.27, 3.28, -1.90), vec3(1.17, 0.07, 1.27))
]

cur_level = levels[7]

def mengerFold(z):
    k1 = tf.min(z.x - z.y, 0.0)
    z.x += k1 * -1.0
    z.y += k1 * 1.0
    k2 = tf.min(z.x - z.z, 0.0)
    z.x += k2 * -1.0
    z.z += k2 * 1.0
    k3 = tf.min(z.y - z.z, 0.0)
    z.y += k3 * -1.0
    z.z += k3 * 1.0

def fractal(p):
    aZ = [tf.sin(cur_level.ang1), tf.cos(cur_level.ang1)]
    aX = [tf.sin(cur_level.ang2), tf.cos(cur_level.ang2)]
    scale = tf.const(1.0)
    orbit = vec3.zero_like(p)
    for i in range(11):
        p = abs(p)
        p.x, p.y = p.x * aZ[1] + p.y * aZ[0], p.x * -aZ[0] + p.y * aZ[1]
        mengerFold(p)
        p.y, p.z = p.y * aX[1] + p.z * aX[0], p.y * -aX[0] + p.z * aX[1]
        p = p * cur_level.scale + cur_level.shift
        scale = scale * cur_level.scale
        orbit = max(orbit, mul(p, cur_level.col))

    return sdBox(p, vec3(6.0, 6.0, 6.0)) / scale, clamp(orbit, 0.0, 1.0)

def map(p):
    return fractal(p)

def calcNormal(p, dx):
    dx = tf.max(dx, 1e-4)
    normal = (vec3(1.0, -1.0, -1.0) * map(p + vec3(1.0, -1.0, -1.0) * dx)[0] +
              vec3(-1.0, -1.0, 1.0) * map(p + vec3(-1.0, -1.0, 1.0) * dx)[0] +
              vec3(-1.0, 1.0, -1.0) * map(p + vec3(-1.0, 1.0, -1.0) * dx)[0] +
              vec3( 1.0,  1.0, 1.0) * map(p + vec3(1.0, 1.0, 1.0) * dx)[0])
    return normalize(normal)
                   
def MarchRay(ro, rd, steps=256):
    td = tf.zeros([])
    def loop_body(k):
        sdf = map(ro + rd * td)[0]
        td.val += sdf
        tf.if_cond((sdf < min_angle * td) | (td > max_depth), lambda: tf.break_loop())

    tf.loop(loop_body, 0, steps, 1)
    return td

light_dir = vec3(0.577, 0.577, 0.577)
focal_length = 1.0

def get_ray(u, v, camera, shape):
    pos = vec3.zero(shape)
    dir = vec3.zero(shape)
    pos.set(vec3(camera[0, 0], camera[0, 1], camera[0, 2]))
    cam_f = vec3(camera[3, 0], camera[3, 1], camera[3, 2])
    cam_u = vec3(camera[2, 0], camera[2, 1], camera[2, 2])
    cam_v = vec3(camera[1, 0], camera[1, 1], camera[1, 2])
    dir.set(cam_f * focal_length + cam_u * u + cam_v * v)
    return pos, normalize(dir)

def project(p, camera):
    dpos = p - vec3(camera[0, 0], camera[0, 1], camera[0, 2])
    cam_f = vec3(camera[3, 0], camera[3, 1], camera[3, 2])
    cam_u = vec3(camera[2, 0], camera[2, 1], camera[2, 2])
    cam_v = vec3(camera[1, 0], camera[1, 1], camera[1, 2])
    nor = focal_length / dot(dpos, cam_f)
    u = dot(dpos, cam_u) * nor
    v = dot(dpos, cam_v) * nor
    return u, v

def CubicHermit(x):
    x2 = x * x
    x3 = x2 * x
    return [-0.5 * x3 + x2 - 0.5 * x, 1.5 * x3 - 2.5 * x2 + 1.0, -1.5 * x3 + 2.0 * x2 + 0.5 * x, 0.5 * x3 - 0.5 * x2]

def CubicIterpCH(tex, x, y, ch):
    xi, yi = tf.floor(x), tf.floor(y)
    xf, yf = x-xi, y-yi
    xi, yi = tf.int(xi), tf.int(yi)

    wx = CubicHermit(xf)
    wy = CubicHermit(yf)

    valueY = 0.0
    for j in range(-1, 3):
        valueX = 0.0
        for i in range(-1, 3):
            valueX = valueX + tex[xi + i, yi + j, ch] * wx[i + 1]
        valueY = valueY + valueX * wy[j + 1]

    return valueY

def BilinearCH(tex, x, y, ch):
    xi, yi = tf.floor(x), tf.floor(y)
    xf, yf = x-xi, y-yi
    xi, yi = tf.int(xi), tf.int(yi)
    oxf, oyf = 1.0-xf, 1.0-yf
    return tex[xi, yi, ch] * oxf * oyf + tex[xi+1, yi, ch] * xf * oyf + tex[xi, yi+1, ch] * oxf * yf + tex[xi+1, yi+1, ch] * xf * yf

def smoothstep(a, b, x):
    t = tf.clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def ray_marcher():
    N, M = H, W
    prev_frame = tf.input([N, M, 3], tf.float32)
    prev_depth = tf.input([N, M], tf.float32)
    environment_map = tf.input([-1, -1, 3], tf.float32)
    env_shape = environment_map.shape

    def sample_background(dir):
        u = 0.5 + tf.atan2(dir.x, dir.z) / (2.0 * np.pi)
        v = 0.5 + tf.asin(dir.y) / np.pi
        pix_i = v * tf.float(env_shape[0] - 1)
        pix_j = u * tf.float(env_shape[1] - 1)
        return vec3(BilinearCH(environment_map, pix_i, pix_j, 0), BilinearCH(environment_map, pix_i, pix_j, 1), BilinearCH(environment_map, pix_i, pix_j, 2))

    # Camera parameters (pos, cam_axis_x, cam_axis_y, cam_axis_z)
    camera = tf.input([4,3], tf.float32)
    prevcamera = tf.input([4,3], tf.float32)
    frame_id = tf.input([1], tf.int32)[0]
   
    canvas = tf.buffer([N, M, 3], tf.float32)
    depth_buffer = tf.buffer([N, M], tf.float32)
    i, j = tf.indices([N, M])
    min_res = tf.min(N, M)

    def ij_to_uv(i, j):
        v = (tf.float(i) - 0.5 * tf.float(N)) / tf.float(min_res)
        u = (tf.float(j) - 0.5 * tf.float(M)) / tf.float(min_res)
        return u, v
    
    def uv_to_ij(u, v):
        i = v * tf.float(min_res) + 0.5 * tf.float(N)
        j = u * tf.float(min_res) + 0.5 * tf.float(M)
        return i, j

    seed = tf.uint(i + j * M + frame_id * N * M)

    def rand():
        seed.val = tf.pcg(seed)
        return tf.float(seed) / 4294967296.0
    
    def udir():
        rx = 2.0 * np.pi * rand()
        ry = tf.acos(2.0 * rand() - 1.0)
        cx, sx = tf.cos(rx), tf.sin(rx)
        cy, sy = tf.cos(ry), tf.sin(ry)
        return vec3(cx * sy, sx * sy, cy)
    
    def normal_rand():
        #box-muller transform
        u1 = rand()
        u2 = rand()
        r = tf.sqrt(-2.0 * tf.log(u1))
        theta = 2.0 * np.pi * u2
        return r * tf.cos(theta), r * tf.sin(theta)

    def hemisphere_dir(normal):
        return normalize(normal + udir())
    
    def random_reflection(normal, dir, roughness):
        rx, ry = normal_rand()
        rz, rw = normal_rand()
        random_normal = normal + roughness * vec3(rx, ry, rz)
        return normalize(reflect(dir, random_normal))
    
    u, v = ij_to_uv(i, j)
    u_off = rand() / tf.float(M)
    v_off = rand() / tf.float(N)
    
    cam_pos, cam_dir = get_ray(u + u_off, v + v_off, camera, i.shape)

    ro = vec3.copy(cam_pos)
    rd = vec3.copy(cam_dir)

    emis = vec3.zero(i.shape)
    atten = vec3.const(1.0, i.shape)
    first_depth = tf.const(0.0)

    def path_tracing_iteration(bounce):
        td = MarchRay(ro, rd)
        hit = ro + rd * td
        tf.if_cond(bounce == 0, lambda: first_depth.set(td))

        def if_body():
            dx = td*min_angle
            norm = calcNormal(hit, dx)
            sdf, col = map(hit)
            col = clamp(col, 0.0, 1.0)
            
            #shooting shadow ray
            new_hit = hit + norm * dx
            light_dir_sph = light_dir + udir() * 0.03
            shadow_td = MarchRay(new_hit, light_dir_sph, 48)
            shadow = tf.select(shadow_td >= max_depth, 1.0, 0.0)
            illum = col * shadow * tf.max(dot(norm, light_dir), 0.0)
            emis.set(emis + mul(atten, illum))
            atten.set(mul(atten, col))

            #next ray
            rd.set(hemisphere_dir(norm))
            #rd.set(random_reflection(norm, rd, 0.25))
            ro.set(new_hit)

        def else_body():
            sky = sample_background(rd)
            emis.set(emis + mul(atten, sky))
            tf.break_loop()
        
        tf.if_cond(td < max_depth, if_body)
        tf.if_cond(td >= max_depth, else_body)

    tf.loop(path_tracing_iteration, 0, 3, 1)
    

    final_color = emis ** (1.0 / 2.2)

    #find previous frame color
    cam_pos, cam_dir = get_ray(u, v, camera, i.shape)
    first_hit = cam_pos + cam_dir * first_depth
    u, v = project(first_hit, prevcamera)
    pi, pj = uv_to_ij(u, v)

    reject = (pi < 0.0) | (pi >= tf.float(N)) | (pj < 0.0) | (pj >= tf.float(M))
    pixi, pixj = tf.int(tf.round(pi)), tf.int(tf.round(pj))
    prev_first_depth = prev_depth[pixi, pixj]
    prev_ro, prev_rd = get_ray(u, v, prevcamera, i.shape)
    prev_hit = prev_ro + prev_rd * prev_first_depth
    ang_distance = distance(normalize(prev_hit - cam_pos),normalize(first_hit - cam_pos))
    accum = tf.select(reject, 0.0, 0.96) * smoothstep(3e-4, 1e-4, ang_distance)
    canvas[i, j, 0] = tf.lerp(final_color.x, CubicIterpCH(prev_frame, pi, pj, 0), accum)
    canvas[i, j, 1] = tf.lerp(final_color.y, CubicIterpCH(prev_frame, pi, pj, 1), accum)
    canvas[i, j, 2] = tf.lerp(final_color.z, CubicIterpCH(prev_frame, pi, pj, 2), accum)
    depth_buffer[i, j] = first_depth

    return [canvas, depth_buffer]

raymarch = tf.compile(ray_marcher)

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

def axis_angle_quaternion(axis, angle):
    """Create a quaternion from an axis and an angle."""
    axis = normalize(axis)
    angle /= 2
    return np.array([np.cos(angle), axis[0] * np.sin(angle), axis[1] * np.sin(angle), axis[2] * np.sin(angle)])

class Camera:
    def __init__(self, position, quaternion):
        self.position = np.array(position, dtype=np.float32)
        self.quaternion = np.array(quaternion, dtype=np.float32)

    def axis(self, axis):
        return quaternion_to_matrix(self.quaternion)[axis, :]

    def move_axis(self, axis, distance):
        self.position += normalize(self.axis(axis)) * distance

    def rotate_axis(self, axis, angle):
        self.quaternion = quaternion_multiply(self.quaternion, axis_angle_quaternion(self.axis(axis), angle))
    
    def get_camera_matrix(self):
        """Get the camera matrix."""
        return np.stack([self.position, *quaternion_to_matrix(self.quaternion)])

def render_image(img, depth, envmap, camera, prev_camera, frame_id):
    camera_matrix_tf = tf.tensor(camera)
    prev_camera_matrix_tf = tf.tensor(prev_camera)
    frame_id_tf = tf.tensor(np.array([frame_id], dtype=np.int32))
    img, depth = raymarch(img, depth, envmap, camera_matrix_tf, prev_camera_matrix_tf, frame_id_tf)
    return img, depth

tf.show_window(W, H, "Path Tracer")

camera = Camera([0, 6.5, -2], axis_angle_quaternion([0, 0, 1], -np.pi/2))
pmx, pmy = tf.get_mouse_position()

angular_speed = 0.005
camera_speed = 0.01
prev_cam_mat = camera.get_camera_matrix()
img = tf.tensor(np.zeros((H, W, 3), dtype=np.float32))
depth = tf.tensor(np.zeros((H, W), dtype=np.float32))
frame_id = 0

#load a hdr environment map using imageio
envmap = imageio.imread(os.path.join(current_dir, "garden_smol.hdr"))
envmap = np.flipud(envmap)
envmap = 0.3*envmap / np.max(envmap)
envmap = np.array(envmap, dtype=np.float32)
envmap = tf.tensor(envmap)

while not tf.window_should_close():
    mx, my = tf.get_mouse_position()

    if tf.is_mouse_button_pressed(tf.MOUSE_BUTTON_0):
        camera.rotate_axis(0, (mx - pmx) * angular_speed)
        camera.rotate_axis(1, (my - pmy) * angular_speed)

    if tf.is_key_pressed(tf.KEY_W):
        camera.move_axis(2, camera_speed)
    if tf.is_key_pressed(tf.KEY_S):
        camera.move_axis(2, -camera_speed)

    if tf.is_key_pressed(tf.KEY_A):
        camera.move_axis(1, -camera_speed)
    if tf.is_key_pressed(tf.KEY_D):
        camera.move_axis(1, camera_speed)

    if tf.is_key_pressed(tf.KEY_Q):
        camera.rotate_axis(2, angular_speed*2)
    if tf.is_key_pressed(tf.KEY_E):
        camera.rotate_axis(2, -angular_speed*2)

    cam_mat = camera.get_camera_matrix()
    img, depth = render_image(img, depth, envmap, cam_mat, prev_cam_mat, frame_id)
    tf.render_frame(img)
    
    pmx, pmy = mx, my
    prev_cam_mat = cam_mat
    frame_id += 1
    
