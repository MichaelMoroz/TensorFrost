import numpy as np
import TensorFrost as tf

_SLANG = r"""
struct FillParams {
    float4 color;
};

[[vk::push_constant]]
FillParams gParams;

[[vk::binding(0,0)]] RWStructuredBuffer<uint> Pixels : register(u0, space0);

[numthreads(64,1,1)]
void csMain(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= Pixels.length()) return;

    float4 c = saturate(gParams.color);
    uint r = (uint)round(c.r * 255.0);
    uint g = (uint)round(c.g * 255.0);
    uint b = (uint)round(c.b * 255.0);
    uint a = (uint)round(c.a * 255.0);
    Pixels[idx] = r | (g << 8) | (b << 16) | (a << 24);
}
"""


def main() -> None:
    width = height = 512
    local_size = 64
    thread_count = width * height
    group_count = max((thread_count + local_size - 1) // local_size, 1)

    window = tf.createWindow(width, height, "TensorFrost Debug Fill")
    pixel_buffer = tf.createBuffer(thread_count, 4, False)
    program = tf.createComputeProgramFromSlang(
        "debug_fill",
        _SLANG,
        "csMain",
        ro_count=0,
        rw_count=1,
        push_constant_size=16,
    )

    color = np.array([0.15, 0.45, 0.95, 1.0], dtype=np.float32)

    while window.isOpen():
        program.run([], [pixel_buffer], group_count, color)
        window.drawBuffer(pixel_buffer, width, height)

    window.close()


if __name__ == "__main__":
    main()
