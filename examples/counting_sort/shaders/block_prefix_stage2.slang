struct BlockPrefixStage2Params {
    uint groupCount;
};

[[vk::push_constant]]
BlockPrefixStage2Params gParams;

[[vk::binding(0,0)]] StructuredBuffer<uint> BlockGroupTotals : register(t0, space0);
[[vk::binding(1,0)]] RWStructuredBuffer<uint> BlockGroupPrefix : register(u1, space0);

[numthreads(64, 1, 1)]
void csBlockPrefixStage2(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0)
        return;

    uint groupCount = gParams.groupCount;
    uint prefix = 0;
    for (uint group = 0; group < groupCount; ++group)
    {
        BlockGroupPrefix[group] = prefix;
        prefix += BlockGroupTotals[group];
    }
}
