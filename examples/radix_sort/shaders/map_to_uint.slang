static const uint TYPE_UINT = 0u;
static const uint TYPE_INT = 1u;
static const uint TYPE_FLOAT = 2u;
static const uint SIGN_BIT = 0x80000000u;
static const uint FULL_MASK = 0xFFFFFFFFu;
static const uint GROUP_SIZE = TF_GROUP_SIZE;

struct MapParams {
    uint count;
    uint typeCode;
};

[[vk::push_constant]]
MapParams gParams;

[[vk::binding(0,0)]] StructuredBuffer<uint> Input : register(t0, space0);
[[vk::binding(1,0)]] RWStructuredBuffer<uint> Output : register(u1, space0);

[numthreads(GROUP_SIZE, 1, 1)]
void csMapToUint(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    uint count = gParams.count;
    if (index >= count)
        return;

    uint typeCode = gParams.typeCode;
    uint value = Input[index];

    if (typeCode == TYPE_INT)
    {
        value ^= SIGN_BIT;
    }
    else if (typeCode == TYPE_FLOAT)
    {
        uint mask = ((value >> 31) == 1u) ? FULL_MASK : SIGN_BIT;
        value ^= mask;
    }

    Output[index] = value;
}
