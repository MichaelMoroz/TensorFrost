static const uint GROUP_SIZE = 128;
static const uint QUARTER_SIZE = GROUP_SIZE / 4;
static const uint MAX_HIST_SIZE = 256;

uint packedCount(uint histogramSize)
{
    return (histogramSize + 3u) >> 2;
}

[[vk::binding(0,0)]] StructuredBuffer<uint> Params : register(t0, space0);
[[vk::binding(1,0)]] StructuredBuffer<uint> PackedHistogram : register(t1, space0);
[[vk::binding(2,0)]] RWStructuredBuffer<uint> GroupHistogram : register(u2, space0);

[numthreads(64, 1, 1)]
void csUnpack(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint histogramSize = Params[1];
    uint numGroups = Params[4];
    uint packedCountLocal = packedCount(histogramSize);

    uint total = numGroups * histogramSize;
    uint index = dispatchThreadID.x;
    if (index >= total)
        return;

    uint group = index / histogramSize;
    uint bit = index % histogramSize;

    uint packedIndex = group * packedCountLocal + (bit >> 2);
    uint packed = PackedHistogram[packedIndex];
    uint shift = (bit & 3u) * 8u;
    uint count = (packed >> shift) & 0xFFu;
    GroupHistogram[index] = count;
}
