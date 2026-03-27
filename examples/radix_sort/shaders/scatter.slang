static const uint GROUP_SIZE = TF_GROUP_SIZE;
static const uint QUARTER_SIZE = GROUP_SIZE / 4;
static const uint HISTOGRAM_SIZE = TF_HISTOGRAM_SIZE;

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

[[vk::binding(0,0)]] StructuredBuffer<uint> KeysIn : register(t0, space0);
[[vk::binding(1,0)]] StructuredBuffer<uint> ValuesIn : register(t1, space0);
[[vk::binding(2,0)]] StructuredBuffer<uint> GroupPrefix : register(t2, space0);
[[vk::binding(3,0)]] StructuredBuffer<uint> BucketScan : register(t3, space0);
[[vk::binding(4,0)]] RWStructuredBuffer<uint> KeysOut : register(u4, space0);
[[vk::binding(5,0)]] RWStructuredBuffer<uint> ValuesOut : register(u5, space0);

groupshared uint tempBits[GROUP_SIZE];
groupshared uint halfCount[HISTOGRAM_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void csScatter(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    uint elementCount = gParams.elementCount;
    uint shift = gParams.shift;
    uint mask = gParams.mask;
    uint numGroups = gParams.numGroups;
    uint hasValues = gParams.hasValues;

    uint group = groupID.x;
    if (group >= numGroups)
        return;

    uint lane = localID.x;

    [unroll] for (uint idx = lane; idx < HISTOGRAM_SIZE; idx += GROUP_SIZE)
    {
        if (idx < HISTOGRAM_SIZE) halfCount[idx] = 0;
    }

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

    if (active) {
        uint prevBucket = (bit == 0) ? 0 : BucketScan[bit - 1u];
        uint prevGroup = (group == 0) ? 0 : GroupPrefix[(group - 1u) * HISTOGRAM_SIZE + bit];

        uint quarterOffset = 0;
        if (quarterIndex > 0)
        {
            uint packed = halfCount[bit];
            uint shiftBytes = (quarterIndex - 1u) * 8u;
            quarterOffset = (packed >> shiftBytes) & 0xFFu;
        }

        uint begin = quarterIndex * QUARTER_SIZE;
        uint localCount = 0;
        [unroll] for (uint t = begin; t < lane; ++t)
        {
            localCount += (tempBits[t] == bit) ? 1u : 0u;
        }

        uint totalOffset = prevBucket + prevGroup + quarterOffset + localCount;
        KeysOut[totalOffset] = key;

        if (hasValues != 0)
            ValuesOut[totalOffset] = value;
    }
}
