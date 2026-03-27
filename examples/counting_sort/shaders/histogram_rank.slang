static const uint GROUP_SIZE = CS_GROUP_SIZE;

struct HistogramParams {
    uint elementCount;
    uint valueCount;
};

[[vk::push_constant]]
HistogramParams gParams;

[[vk::binding(0,0)]] StructuredBuffer<uint> KeysIn : register(t0, space0);
[[vk::binding(1,0)]] RWStructuredBuffer<uint> Histogram : register(u1, space0);
[[vk::binding(2,0)]] RWStructuredBuffer<uint> LocalRank : register(u2, space0);

[numthreads(GROUP_SIZE, 1, 1)]
void csHistogramAndRank(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    if (index >= gParams.elementCount)
        return;

    uint value = KeysIn[index];
    if (value >= gParams.valueCount)
        return;

    uint rank = 0;
    InterlockedAdd(Histogram[value], 1u, rank);
    LocalRank[index] = rank;
}
