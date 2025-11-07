static const uint GROUP_SIZE = 128;
static const uint QUARTER_SIZE = GROUP_SIZE / 4;
static const uint MAX_HIST_SIZE = 256;

uint packedCount(uint histogramSize)
{
    return (histogramSize + 3u) >> 2;
}

[[vk::binding(0,0)]] StructuredBuffer<uint> Params : register(t0, space0);
[[vk::binding(1,0)]] StructuredBuffer<uint> BlockTotals : register(t1, space0);
[[vk::binding(2,0)]] RWStructuredBuffer<uint> BlockPrefix : register(u2, space0);

[numthreads(64, 1, 1)]
void csPrefixBlocks(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint histogramSize = Params[1];
    uint blockCount = Params[6];
    uint bucket = dispatchThreadID.x;
    if (bucket >= histogramSize)
        return;

    uint sum = 0;
    for (uint block = 0; block < blockCount; ++block)
    {
        uint idx = block * histogramSize + bucket;
        uint total = BlockTotals[idx];
        BlockPrefix[idx] = sum;
        sum += total;
    }
}
