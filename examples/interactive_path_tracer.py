import TensorFrost as tf
import numpy as np
import time

tf.initialize(tf.opengl)

W = 1920
H = 1080
eps = 0.0001
m_pow = 8.0
max_depth = 50.0
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

def normalize(a):
    return a / length(a)

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

def sky_color(dir):
    fsun = vec3(0.577, 0.577, 0.577)
    Br = 0.0025
    Bm = 0.0003
    g = 0.9800
    nitrogen = vec3(0.650, 0.570, 0.475)
    Kr = Br * nitrogen ** -4.0
    Km = Bm * nitrogen ** -0.84

    brightnees = tf.exp(-tf.sqrt(tf.pow(tf.abs(tf.min(5.0 * (dir.y - 0.1), 0.0)), 2.0) + 0.1))
    dir.y = tf.max(dir.y, 0.0)
    dir = normalize(dir)

    mu = dot(normalize(dir), normalize(fsun))
    exp1 = tf.exp(-((dir.y + fsun.y * 4.0) * (tf.exp(-dir.y * 16.0) + 0.1) / 80.0) / Br) * (tf.exp(-dir.y * 16.0) + 0.1)
    exp2 = exp(- Kr * exp1 / Br)
    val1 = exp2 * tf.exp(-dir.y * tf.exp(-dir.y * 8.0) * 4.0) * tf.exp(-dir.y * 2.0) * 4.0
    val2 = (1.0 - tf.exp(fsun.y)) * 0.2
    extinction = lerp(val1, vec3(val2,val2,val2), -fsun.y * 0.2 + 0.5)
    b0 = (1.0 - g * g) / (2.0 + g * g) / tf.pow(1.0 + g * g - 2.0 * g * mu, 1.5)
    b1 = (Kr + Km * b0) * brightnees * 3.0 / (8.0 * 3.14) * (1.0 + mu * mu) / (Br + Bm)
    sky_col = vec3(extinction.x * b1.x, extinction.y * b1.y, extinction.z * b1.z)
    sky_col = 0.4 * clamp(sky_col, 0.0, 1000.0)
    return sky_col ** 1.5

def mandelbulb(p):
    w = vec3.copy(p)
    col = vec3.copy(abs(w))
    m = dot(w,w)
    dz = tf.const(1.0)
    def loop_body(i):
        dz.val = m_pow * m ** (0.5*(m_pow - 1.0)) * dz + 1.0
        r = length(w)+1e-6
        b = m_pow * tf.acos(w.y/r)
        a = m_pow * tf.atan2(w.x, w.z)
        c = r ** m_pow
        w.set(vec3(p.x + c * tf.sin(b) * tf.sin(a), p.y + c * tf.cos(b), p.z + c * tf.sin(b) * tf.cos(a)))
        col.set(min(col, abs(w)))
        m.val = dot(w,w)
        tf.if_cond(m > 256.0, lambda: tf.break_loop())

    tf.loop(loop_body, 0, 5, 1)
    sdf = 0.25 * tf.log(m) * tf.sqrt(m) / dz
    return sdf, col

def map(p):
    return mandelbulb(p)

def calcNormal(p):
    sdf = map(p)[0]
    sdfx = map(p + vec3(eps, 0.0, 0.0))[0]
    sdfy = map(p + vec3(0.0, eps, 0.0))[0]
    sdfz = map(p + vec3(0.0, 0.0, eps))[0]
    return normalize(vec3(sdfx - sdf, sdfy - sdf, sdfz - sdf))

def MarchRay(ro, rd, steps=256):
    td = tf.zeros([])
    def loop_body(k):
        sdf = map(ro + rd * td)[0]
        td.val += sdf
        tf.if_cond((sdf < min_angle * td) | (td > max_depth), lambda: tf.break_loop())

    tf.loop(loop_body, 0, steps, 1)
    return td

def MarchSoftShadow(ro, rd, w, steps=64):
    td = tf.const(0.0)
    psdf = tf.const(1e10)
    res = tf.const(1.0)
    it = tf.const(0)
    def loop_body(k):
        sdf = map(ro + rd * td)[0]
        y = tf.select(k == 0, 0.0, sdf * sdf / (2.0 * psdf))
        d = tf.sqrt(sdf * sdf - y * y)
        res.val = tf.min(res, d / (w * tf.max(1e-6, td - y)))
        psdf.val = sdf
        td.val += sdf
        it.val = k
        tf.if_cond((td > max_depth) | (res < 0.001), lambda: tf.break_loop())

    tf.loop(loop_body, 0, steps, 1)
    res = tf.select(it == steps - 1, 0.0, tf.clamp(res, 0.0, 1.0))
    return res * res * (3.0 - 2.0 * res)

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

def ray_marcher():
    N, M = H, W
    prev_frame = tf.input([N, M, 3], tf.float32)
    # Camera parameters (pos, cam_axis_x, cam_axis_y, cam_axis_z)
    camera = tf.input([4,3], tf.float32)
    prevcamera = tf.input([4,3], tf.float32)
    frame_id = tf.input([1], tf.int32)[0]
   
    canvas = tf.buffer([N, M, 3], tf.float32)
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
        c = tf.cos(rx)
        s = tf.sin(rx)
        return vec3(c * s, s * s, c)
    
    def hemisphere_dir(normal):
        return normalize(normal + udir())
    
    u, v = ij_to_uv(i, j)
    
    ro, rd = get_ray(u, v, camera, i.shape)

    emis = vec3.zero(i.shape)
    atten = vec3.const(1.0, i.shape)
    #first_depth = tf.const(0.0)
    first_hit = vec3.zero(i.shape)

    def path_tracing_iteration(bounce):
        td = MarchRay(ro, rd)
        hit = ro + rd * td
        #tf.if_cond(bounce == 0, lambda: first_depth.set(td))
        tf.if_cond(bounce == 0, lambda: first_hit.set(hit))

        def if_body():
            norm = calcNormal(hit)
            sdf, col = map(hit)
            col = clamp(col, 0.9, 1.0)
            
            #shooting shadow ray
            shadow = MarchSoftShadow(hit, light_dir, 0.05)
            illum = col * shadow * tf.max(dot(norm, light_dir), 0.0)
            emis.set(emis + mul(atten, illum))
            atten.set(mul(atten, col))

            #next ray
            rd.set(hemisphere_dir(norm))
            ro.set(hit + norm * 0.01)

        def else_body():
            sky = sky_color(rd)
            emis.set(emis + mul(atten, sky))
            tf.break_loop()
        
        tf.if_cond(td < max_depth, if_body)
        tf.if_cond(td >= max_depth, else_body)

    tf.loop(path_tracing_iteration, 0, 3, 1)
    

    final_color = emis ** (1.0 / 2.2)

    #find previous frame color
    u, v = project(first_hit, prevcamera)
    pi, pj = uv_to_ij(u, v)
    
    accum = 0.9
    canvas[i, j, 0] = tf.lerp(final_color.x, CubicIterpCH(prev_frame, pi, pj, 0), accum)
    canvas[i, j, 1] = tf.lerp(final_color.y, CubicIterpCH(prev_frame, pi, pj, 1), accum)
    canvas[i, j, 2] = tf.lerp(final_color.z, CubicIterpCH(prev_frame, pi, pj, 2), accum)
    

    return [canvas]

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

def render_image(img, camera, prev_camera, frame_id):
    camera_matrix_tf = tf.tensor(camera)
    prev_camera_matrix_tf = tf.tensor(prev_camera)
    frame_id_tf = tf.tensor(np.array([frame_id], dtype=np.int32))
    img, = raymarch(img, camera_matrix_tf, prev_camera_matrix_tf, frame_id_tf)
    return img

tf.show_window(W, H, "Path Tracer")

camera = Camera([0, 0, -2], axis_angle_quaternion([0, 0, 1], -np.pi/2))
pmx, pmy = tf.get_mouse_position()

angular_speed = 0.005
camera_speed = 0.005
prev_cam_mat = camera.get_camera_matrix()
img = tf.tensor(np.zeros((H, W, 3), dtype=np.float32))
frame_id = 0

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
    img = render_image(img, cam_mat, prev_cam_mat, frame_id)
    tf.render_frame(img)
    
    pmx, pmy = mx, my
    prev_cam_mat = cam_mat
    frame_id += 1
    
