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

[[vk::binding(0,0)]] StructuredBuffer<uint> GroupPrefix : register(t0, space0);
[[vk::binding(1,0)]] RWStructuredBuffer<uint> BucketScan : register(u1, space0);

[numthreads(64, 1, 1)]
void csBucketScan(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0)
        return;

    uint histogramSize = gParams.histogramSize;
    uint numGroups = gParams.numGroups;

    if (histogramSize == 0)
        return;

    if (numGroups == 0)
    {
        for (uint bit = 0; bit < histogramSize; ++bit)
        {
            BucketScan[bit] = 0;
        }
        return;
    }

    uint base = (numGroups - 1u) * histogramSize;
    uint sum = 0;
    for (uint bit = 0; bit < histogramSize; ++bit)
    {
        uint total = GroupPrefix[base + bit];
        sum += total;
        BucketScan[bit] = sum;
    }
}
