uint packedCount(uint histogramSize)
{
    return (histogramSize + 3u) >> 2;
}

struct SortParams {
    uint elementCount;
    uint histogramSize;
    uint shift;
    uint mask;
    uint numGroups;
    uint blockSize;
    uint blockCount;
    uint hasValues;
};

[[vk::push_constant]]
SortParams gParams;

[[vk::binding(0,0)]] StructuredBuffer<uint> PackedHistogram : register(t0, space0);
[[vk::binding(1,0)]] RWStructuredBuffer<uint> GroupHistogram : register(u1, space0);

[numthreads(64, 1, 1)]
void csUnpack(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint histogramSize = gParams.histogramSize;
    uint numGroups = gParams.numGroups;
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
