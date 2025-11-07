static const uint TYPE_UINT = 0u;
static const uint TYPE_INT = 1u;
static const uint TYPE_FLOAT = 2u;
static const uint SIGN_BIT = 0x80000000u;
static const uint FULL_MASK = 0xFFFFFFFFu;

[[vk::binding(0,0)]] StructuredBuffer<uint> Params : register(t0, space0);
[[vk::binding(1,0)]] StructuredBuffer<uint> Input : register(t1, space0);
[[vk::binding(2,0)]] RWStructuredBuffer<uint> Output : register(u2, space0);

[numthreads(128, 1, 1)]
void csMapFromUint(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    uint count = Params[0];
    if (index >= count)
        return;

    uint typeCode = Params[1];
    uint value = Input[index];

    if (typeCode == TYPE_INT)
    {
        value ^= SIGN_BIT;
    }
    else if (typeCode == TYPE_FLOAT)
    {
        uint mask = ((value >> 31) == 0u) ? FULL_MASK : SIGN_BIT;
        value ^= mask;
    }

    Output[index] = value;
}
