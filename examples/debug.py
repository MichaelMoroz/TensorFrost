import numpy as np
import TensorFrost as tf

# GLSL: 1D dispatch (local_size_x=64). Pixels are packed with packUnorm4x8.
glsl = r"""
#version 450
layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer Params { float p[]; }; // [w,h,xmin,ymin,dx,dy,maxIter,isBGRA]
layout(std430, binding = 1) writeonly buffer Pixels { uint out_u32[]; };

vec3 palette(float t) {
    // simple smooth palette
    return vec3(0.5 + 0.5*cos(6.28318*(vec3(0.0,0.33,0.67)+t)));
}

void main() {
    uint idx1D = gl_GlobalInvocationID.x;
    int W = int(p[0] + 0.5), H = int(p[1] + 0.5);
    uint N = uint(W*H);
    if (idx1D >= N) return;

    int x = int(idx1D % uint(W));
    int y = int(idx1D / uint(W));

    float xmin = p[2], ymin = p[3], dx = p[4], dy = p[5];
    int maxIter = int(p[6] + 0.5);
    bool isBGRA = (p[7] > 0.5);

    float cx = xmin + float(x) * dx;
    float cy = ymin + float(y) * dy;

    float zx = 0.0, zy = 0.0;
    int i = 0;
    for (; i < maxIter; ++i) {
        float zx2 = zx*zx - zy*zy + cx;
        float zy2 = 2.0*zx*zy + cy;
        zx = zx2; zy = zy2;
        if (zx*zx + zy*zy > 4.0) break;
    }

    float t = (i == maxIter) ? 0.0 :
              float(i) - log2(log(length(vec2(zx,zy)))) + 4.0;
    t = clamp(t / float(maxIter), 0.0, 1.0);

    vec3 rgb = palette(t);
    vec4 c = vec4(rgb, 1.0);
    uint packed = isBGRA ? packUnorm4x8(c.bgra) : packUnorm4x8(c);
    out_u32[idx1D] = packed;
}
"""

def main():
    W, H = 1024, 768
    local_size = 64
    group_count = max((W * H + local_size - 1) // local_size, 1)
    win = tf.createWindow(W, H, "Mandelbrot (compute → buffer → swapchain)")
    fmt = int(win.format)
    is_bgra = fmt in (44, 50)  # VK_FORMAT_B8G8R8A8_UNORM / _SRGB

    pix = tf.createBuffer(W*H, 4, False)        # uint32 pixels
    params = tf.createBuffer(8, 4, True)        # 8 float32 params

    prog = tf.createComputeProgramFromGLSL(glsl, ro_count=1, rw_count=1)

    # view rectangle with aspect correction
    xspan = 3.0
    yspan = xspan * (H / float(W))
    xmin, ymin = -2.0, -yspan * 0.5
    dx, dy = xspan / W, yspan / H
    max_iter = 500.0

    p = np.array([float(W), float(H), xmin, ymin, dx, dy, max_iter, 1.0 if is_bgra else 0.0], dtype=np.float32)
    params.setData(p)

    while win.isOpen():
        prog.run([params], [pix], group_count)
        win.drawBuffer(pix, W, H)

    win.close()

if __name__ == "__main__":
    main()