static const uint GROUP_SIZE = CS_GROUP_SIZE;

struct ScatterParams {
    uint elementCount;
    uint valueCount;
    uint blockSpan;
    uint blocksPerGroup;
    uint hasValues;
};

[[vk::push_constant]]
ScatterParams gParams;

[[vk::binding(0,0)]] StructuredBuffer<uint> KeysIn : register(t0, space0);
[[vk::binding(1,0)]] StructuredBuffer<uint> ValuesIn : register(t1, space0);
[[vk::binding(2,0)]] StructuredBuffer<uint> LocalRank : register(t2, space0);
[[vk::binding(3,0)]] StructuredBuffer<uint> LocalPrefix : register(t3, space0);
[[vk::binding(4,0)]] StructuredBuffer<uint> BlockPrefixStage1 : register(t4, space0);
[[vk::binding(5,0)]] StructuredBuffer<uint> BlockGroupPrefix : register(t5, space0);
[[vk::binding(6,0)]] RWStructuredBuffer<uint> KeysOut : register(u6, space0);
[[vk::binding(7,0)]] RWStructuredBuffer<uint> ValuesOut : register(u7, space0);

[numthreads(GROUP_SIZE, 1, 1)]
void csScatter(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    uint elementCount = gParams.elementCount;
    if (index >= elementCount)
        return;

    uint value = KeysIn[index];
    if (value >= gParams.valueCount)
        return;

    uint rank = LocalRank[index];
    uint block = value / gParams.blockSpan;
    uint group = block / max(gParams.blocksPerGroup, 1u);
    uint localOffset = LocalPrefix[value];
    uint blockOffset = BlockPrefixStage1[block] + BlockGroupPrefix[group];
    uint destination = blockOffset + localOffset + rank;

    KeysOut[destination] = value;
    if (gParams.hasValues != 0)
    {
        ValuesOut[destination] = ValuesIn[index];
    }
}
