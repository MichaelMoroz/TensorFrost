struct SegmentScanParams {
    uint segmentCount;
    uint segmentSpan;
    uint inputLimit;
};

[[vk::push_constant]]
SegmentScanParams gParams;

[[vk::binding(0,0)]] StructuredBuffer<uint> InputData : register(t0, space0);
[[vk::binding(1,0)]] RWStructuredBuffer<uint> PrefixOutput : register(u1, space0);
[[vk::binding(2,0)]] RWStructuredBuffer<uint> SegmentTotals : register(u2, space0);

[numthreads(64, 1, 1)]
void csSegmentScan(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint segment = dispatchThreadID.x;
    if (segment >= gParams.segmentCount)
        return;

    uint start = segment * gParams.segmentSpan;
    if (start >= gParams.inputLimit)
    {
        SegmentTotals[segment] = 0;
        return;
    }

    uint endIndex = min(start + gParams.segmentSpan, gParams.inputLimit);
    uint prefix = 0;
    for (uint idx = start; idx < endIndex; ++idx)
    {
        PrefixOutput[idx] = prefix;
        prefix += InputData[idx];
    }

    SegmentTotals[segment] = prefix;
}
