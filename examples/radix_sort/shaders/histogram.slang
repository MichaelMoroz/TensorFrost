static const uint GROUP_SIZE = TF_GROUP_SIZE;
static const uint QUARTER_SIZE = GROUP_SIZE / 4;
static const uint MAX_HIST_SIZE = 256u;

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

[[vk::binding(0,0)]] StructuredBuffer<uint> KeysIn : register(t0, space0);
[[vk::binding(1,0)]] RWStructuredBuffer<uint> PackedHistogram : register(u1, space0);

groupshared uint sPacked[MAX_HIST_SIZE / 4];

[numthreads(GROUP_SIZE, 1, 1)]
void csHistogram(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    uint elementCount = gParams.elementCount;
    uint histogramSize = gParams.histogramSize;
    uint shift = gParams.shift;
    uint mask = gParams.mask;
    uint numGroups = gParams.numGroups;

    if (histogramSize > MAX_HIST_SIZE)
        return;

    uint group = groupID.x;
    if (group >= numGroups)
        return;

    uint packedCountLocal = packedCount(histogramSize);
    uint lane = localID.x;

    for (uint idx = lane; idx < packedCountLocal; idx += GROUP_SIZE)
    {
        sPacked[idx] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint globalIndex = group * GROUP_SIZE + lane;
    if (globalIndex < elementCount)
    {
        uint key = KeysIn[globalIndex];
        uint bit = (key >> shift) & mask;
        uint slot = bit >> 2;
        if (slot < packedCountLocal)
        {
            uint laneShift = (bit & 3u) * 8u;
            InterlockedAdd(sPacked[slot], 1u << laneShift);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint idx = lane; idx < packedCountLocal; idx += GROUP_SIZE)
    {
        PackedHistogram[group * packedCountLocal + idx] = sPacked[idx];
    }
}
