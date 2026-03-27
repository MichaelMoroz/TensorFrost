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

[[vk::binding(0,0)]] StructuredBuffer<uint> BlockTotals : register(t0, space0);
[[vk::binding(1,0)]] RWStructuredBuffer<uint> BlockPrefix : register(u1, space0);

[numthreads(64, 1, 1)]
void csPrefixBlocks(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint histogramSize = gParams.histogramSize;
    uint blockCount = gParams.blockCount;
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
