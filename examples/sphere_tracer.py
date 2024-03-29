import TensorFrost as tf
import numpy as np
import time

tf.initialize(tf.opengl)

S = 1024
eps = 0.005
m_pow = 8.0
max_depth = 6.0
min_dist = 0.001

def mandelbulb(px, py, pz):
    wx, wy, wz = tf.zeros(px.shape, tf.float32), tf.zeros(px.shape, tf.float32), tf.zeros(px.shape, tf.float32)
    wx.set(px), wy.set(py), wz.set(pz)
    m = wx*wx + wy*wy + wz*wz
    dz = tf.zeros(px.shape, tf.float32)
    dz.set(1.0)
    #orbit trap
    cx, cy, cz = tf.zeros(px.shape, tf.float32), tf.zeros(px.shape, tf.float32), tf.zeros(px.shape, tf.float32)
    cx.set(tf.abs(wx)), cy.set(tf.abs(wy)), cz.set(tf.abs(wz))
    def loop_body(i):
        dz.set(m_pow * m ** (0.5*(m_pow - 1.0)) * dz + 1.0)
        r = tf.sqrt(wx*wx + wy*wy + wz*wz)
        b = m_pow * tf.acos(wy/r)
        a = m_pow * tf.atan2(wx, wz)
        c = r ** m_pow
        wx.set(px + c * tf.sin(b) * tf.sin(a))
        wy.set(py + c * tf.cos(b))
        wz.set(pz + c * tf.sin(b) * tf.cos(a))
        cx.set(tf.min(cx, tf.abs(wx)))
        cy.set(tf.min(cy, tf.abs(wy)))
        cz.set(tf.min(cz, tf.abs(wz)))
        m.set(wx*wx + wy*wy + wz*wz)
        tf.if_cond(m > 256.0, lambda: tf.break_loop())

    tf.loop(loop_body, 0, 4, 1)
    sdf = 0.25 * tf.log(m) * tf.sqrt(m) / dz
    return sdf, cx, cy, cz

def calcNormal(px, py, pz):
    sdf = mandelbulb(px, py, pz)[0]
    sdfx = mandelbulb(px + eps, py, pz)[0]
    sdfy = mandelbulb(px, py + eps, pz)[0]
    sdfz = mandelbulb(px, py, pz + eps)[0]
    nx = sdfx - sdf
    ny = sdfy - sdf
    nz = sdfz - sdf
    mag = tf.sqrt(nx*nx + ny*ny + nz*nz)
    return nx/mag, ny/mag, nz/mag

def MarchRay(shape, cam, dir, steps=128):
    camx, camy, camz = cam
    dirx, diry, dirz = dir

    td = tf.zeros(shape, tf.float32)
    def loop_body(k):
        px = camx + dirx * td
        py = camy + diry * td
        pz = camz + dirz * td
        sdf = mandelbulb(px, py, pz)[0]
        td.set(td + sdf)
        tf.if_cond((sdf < min_dist) | (td > max_depth), lambda: tf.break_loop())

    tf.loop(loop_body, 0, steps, 1)
    return camx + dirx * td, camy + diry * td, camz + dirz * td, td

light_dir_x = -0.577
light_dir_y = -0.577
light_dir_z = -0.577

def spherical_to_cartesian(r, theta, phi):
    # Convert spherical to Cartesian coordinates
    x = r * tf.sin(theta) * tf.cos(phi)
    y = r * tf.sin(theta) * tf.sin(phi)
    z = r * tf.cos(theta)
    return x, y, z

def cross(a, b):
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def normalize(a):
    mag = tf.sqrt(dot(a, a))
    return a[0] / mag, a[1] / mag, a[2] / mag

def mul(a, b):
    return a[0] * b, a[1] * b, a[2] * b

def add(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]

def camera_axes(r, phi, theta):
    # Camera position
    cam = spherical_to_cartesian(r, theta+1e-4, phi+1e-4)
    
    # Forward vector (normalized vector from camera position to origin)
    forward = mul(normalize(cam), -1.0)
    
    # Assuming Z is up
    world_up = 0.0, 0.0, 1.0
    
    # Right vector (cross product of world up and forward vector)
    right = cross(world_up, forward)
    right = normalize(right)
    
    # Recalculate the up vector to ensure orthogonality
    up = cross(forward, right)
    up = normalize(up)
    
    return cam, up, forward, right

def get_camera(u, v, dist, phi, theta):
    cam, up, forward, right = camera_axes(dist, phi, theta)

    dirx = forward[0] + u * right[0] + v * up[0]
    diry = forward[1] + u * right[1] + v * up[1]
    dirz = forward[2] + u * right[2] + v * up[2]

    # normalize direction
    direction = normalize((dirx, diry, dirz))

    return cam, direction

def ray_marcher():
    camera_params = tf.input([3], tf.float32)
    N, M = S, S
    canvas = tf.zeros([N, M, 3], tf.float32)
    i, j = tf.indices([N, M])
    min_res = tf.min(N, M)
    v, u = tf.float(i), tf.float(j)
    v = (v - 0.5 * tf.float(N)) / tf.float(min_res)
    u = (u - 0.5 * tf.float(M)) / tf.float(min_res)

    cam, dir = get_camera(u, v, camera_params[0], camera_params[1], camera_params[2])
    
    px, py, pz, td = MarchRay(i.shape, cam, dir)

    def if_body():
        # Lighting
        norm = calcNormal(px, py, pz)
        sdf, m, cx, cy = mandelbulb(px, py, pz)
        r, g, b = tf.clamp(m, 0.6, 1.0), tf.clamp(cx, 0.6, 1.0), tf.clamp(cy, 0.6, 1.0)
        col = dot(norm, (light_dir_x, light_dir_y, light_dir_z)) * 0.5 + 0.5
        canvas[i, j, 0] = col * r
        canvas[i, j, 1] = col * g
        canvas[i, j, 2] = col * b

    tf.if_cond(td < max_depth, if_body)
    
    return [canvas]

raymarch = tf.compile(ray_marcher)


def render_mandelbulb(phi, theta):
    cam_params = tf.tensor( np.array([3.0, phi, theta], dtype=np.float32) )

    img, = raymarch(cam_params)

    return img

tf.show_window(S, S, "Sphere tracer")

import time

init_time = time.time()

while not tf.window_should_close():
    mx, my = tf.get_mouse_position()
    cur_time = time.time() - init_time

    phi = mx * 2.0 * np.pi / S
    theta = my * np.pi / S

    img = render_mandelbulb(phi, theta)
    tf.render_frame(img)

tf.hide_window()