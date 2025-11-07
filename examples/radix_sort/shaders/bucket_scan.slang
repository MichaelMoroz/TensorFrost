static const uint GROUP_SIZE = 128;
static const uint QUARTER_SIZE = GROUP_SIZE / 4;
static const uint MAX_HIST_SIZE = 256;

uint packedCount(uint histogramSize)
{
    return (histogramSize + 3u) >> 2;
}

[[vk::binding(0,0)]] StructuredBuffer<uint> Params : register(t0, space0);
[[vk::binding(1,0)]] StructuredBuffer<uint> GroupPrefix : register(t1, space0);
[[vk::binding(2,0)]] RWStructuredBuffer<uint> BucketScan : register(u2, space0);

[numthreads(64, 1, 1)]
void csBucketScan(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0)
        return;

    uint histogramSize = Params[1];
    uint numGroups = Params[4];

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
