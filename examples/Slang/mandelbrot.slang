[[vk::binding(0,0)]] StructuredBuffer<float> Params : register(t0, space0);
[[vk::binding(1,0)]] RWStructuredBuffer<uint> Pixels : register(u1, space0);

static float3 palette(float t) {
    return sin(6.3 * float3(0.0, 0.33, 0.67) * t);
}

[numthreads(64,1,1)]
void csMain(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    int  W = (int)(Params[0] + 0.5);
    int  H = (int)(Params[1] + 0.5);
    uint N = (uint)(W * H);
    if (idx >= N) return;

    int x = int(idx % (uint)W);
    int y = int(idx / (uint)W);

    float xmin = Params[2], ymin = Params[3], dx = Params[4], dy = Params[5];
    int   maxIter = (int)(Params[6] + 0.5);
    bool  isBGRA  = (Params[7] > 0.5);

    float cx = xmin + float(x) * dx;
    float cy = ymin + float(y) * dy;

    float zx = 0.0, zy = 0.0;
    int i = 0;
    [loop]
    for (; i < maxIter; ++i) {
        float zx2 = zx*zx - zy*zy + cx;
        zy = 2.0*zx*zy + cy;
        zx = zx2;
        if (zx*zx + zy*zy > 4.0) break;
    }

    float t = (float(i) - log2(log(sqrt(zx*zx + zy*zy))) + 4.0);
    t = clamp(t / float(maxIter), 0.0, 1.0);

    float3 rgb = palette(t);
    float4 c = float4(rgb, 1.0);

    uint r = (uint)round(saturate(c.r) * 255.0);
    uint g = (uint)round(saturate(c.g) * 255.0);
    uint b = (uint)round(saturate(c.b) * 255.0);
    uint a = (uint)round(saturate(c.a) * 255.0);
    uint packed = isBGRA ? (b | (g<<8) | (r<<16) | (a<<24))
                         : (r | (g<<8) | (b<<16) | (a<<24));
    Pixels[idx] = packed;
}