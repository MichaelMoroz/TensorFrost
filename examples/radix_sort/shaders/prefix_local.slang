static const uint GROUP_SIZE = 128;
static const uint QUARTER_SIZE = GROUP_SIZE / 4;
static const uint MAX_HIST_SIZE = 256;

uint packedCount(uint histogramSize)
{
    return (histogramSize + 3u) >> 2;
}

[[vk::binding(0,0)]] StructuredBuffer<uint> Params : register(t0, space0);
[[vk::binding(1,0)]] StructuredBuffer<uint> GroupHistogram : register(t1, space0);
[[vk::binding(2,0)]] RWStructuredBuffer<uint> GroupPrefix : register(u2, space0);
[[vk::binding(3,0)]] RWStructuredBuffer<uint> BlockTotals : register(u3, space0);

[numthreads(64, 1, 1)]
void csPrefixLocal(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint histogramSize = Params[1];
    uint numGroups = Params[4];
    uint blockSize = Params[5];
    uint blockCount = Params[6];

    uint totalThreads = blockCount * histogramSize;
    uint index = dispatchThreadID.x;
    if (index >= totalThreads)
        return;

    uint block = index / histogramSize;
    uint bit = index % histogramSize;

    uint startGroup = block * blockSize;
    if (startGroup >= numGroups)
    {
        BlockTotals[index] = 0;
        return;
    }

    uint endGroup = min(startGroup + blockSize, numGroups);
    uint sum = 0;
    for (uint g = startGroup; g < endGroup; ++g)
    {
        uint idx = g * histogramSize + bit;
        sum += GroupHistogram[idx];
        GroupPrefix[idx] = sum;
    }
    BlockTotals[index] = sum;
}
