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

[[vk::binding(0,0)]] StructuredBuffer<uint> BlockPrefix : register(t0, space0);
[[vk::binding(1,0)]] RWStructuredBuffer<uint> GroupPrefix : register(u1, space0);

[numthreads(64, 1, 1)]
void csPrefixAccumulate(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint histogramSize = gParams.histogramSize;
    uint numGroups = gParams.numGroups;
    uint blockSize = gParams.blockSize;
    uint blockCount = gParams.blockCount;

    uint totalThreads = blockCount * histogramSize;
    uint index = dispatchThreadID.x;
    if (index >= totalThreads)
        return;

    uint block = index / histogramSize;
    uint bit = index % histogramSize;

    uint startGroup = block * blockSize;
    if (startGroup >= numGroups)
        return;

    uint endGroup = min(startGroup + blockSize, numGroups);
    uint prefix = BlockPrefix[index];
    for (uint g = startGroup; g < endGroup; ++g)
    {
        uint idx = g * histogramSize + bit;
        GroupPrefix[idx] += prefix;
    }
}
