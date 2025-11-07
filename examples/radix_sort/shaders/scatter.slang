static const uint GROUP_SIZE = 128;
static const uint QUARTER_SIZE = GROUP_SIZE / 4;
static const uint MAX_HIST_SIZE = 256;

uint packedCount(uint histogramSize)
{
    return (histogramSize + 3u) >> 2;
}

[[vk::binding(0,0)]] StructuredBuffer<uint> Params : register(t0, space0);
[[vk::binding(1,0)]] StructuredBuffer<uint> KeysIn : register(t1, space0);
[[vk::binding(2,0)]] StructuredBuffer<uint> ValuesIn : register(t2, space0);
[[vk::binding(3,0)]] StructuredBuffer<uint> GroupPrefix : register(t3, space0);
[[vk::binding(4,0)]] StructuredBuffer<uint> BucketScan : register(t4, space0);
[[vk::binding(5,0)]] RWStructuredBuffer<uint> KeysOut : register(u5, space0);
[[vk::binding(6,0)]] RWStructuredBuffer<uint> ValuesOut : register(u6, space0);

groupshared uint tempBits[GROUP_SIZE];
groupshared uint halfCount[MAX_HIST_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void csScatter(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    uint elementCount = Params[0];
    uint histogramSize = Params[1];
    uint shift = Params[2];
    uint mask = Params[3];
    uint numGroups = Params[4];
    uint hasValues = Params[7];

    if (histogramSize > MAX_HIST_SIZE)
        return;

    uint group = groupID.x;
    if (group >= numGroups)
        return;

    uint lane = localID.x;

    for (uint idx = lane; idx < histogramSize; idx += GROUP_SIZE)
    {
        halfCount[idx] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint globalIndex = group * GROUP_SIZE + lane;
    bool active = (globalIndex < elementCount);
    uint bit = 0;
    uint key = 0;
    uint value = 0;

    if (active)
    {
        key = KeysIn[globalIndex];
        bit = (key >> shift) & mask;
        tempBits[lane] = bit;
        if (hasValues != 0)
            value = ValuesIn[globalIndex];
    }
    else
    {
        tempBits[lane] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint quarterIndex = lane / QUARTER_SIZE;
    if (active && quarterIndex < 3)
    {
        uint inc = 0;
        if (quarterIndex < 1) inc |= 1u;
        if (quarterIndex < 2) inc |= (1u << 8);
        if (quarterIndex < 3) inc |= (1u << 16);
        InterlockedAdd(halfCount[bit], inc);
    }
    GroupMemoryBarrierWithGroupSync();

    if (!active)
        return;

    uint prevBucket = (bit == 0) ? 0 : BucketScan[bit - 1u];
    uint prevGroup = (group == 0) ? 0 : GroupPrefix[(group - 1u) * histogramSize + bit];

    uint quarterOffset = 0;
    if (quarterIndex > 0)
    {
        uint packed = halfCount[bit];
        uint shiftBytes = (quarterIndex - 1u) * 8u;
        quarterOffset = (packed >> shiftBytes) & 0xFFu;
    }

    uint begin = quarterIndex * QUARTER_SIZE;
    uint localCount = 0;
    for (uint t = begin; t < lane; ++t)
    {
        if (tempBits[t] == bit)
            ++localCount;
    }

    uint totalOffset = prevBucket + prevGroup + quarterOffset + localCount;
    KeysOut[totalOffset] = key;

    if (hasValues != 0)
        ValuesOut[totalOffset] = value;
}
