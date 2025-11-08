static const uint TYPE_UINT = 0u;
static const uint TYPE_INT = 1u;
static const uint TYPE_FLOAT = 2u;
static const uint SIGN_BIT = 0x80000000u;
static const uint FULL_MASK = 0xFFFFFFFFu;
static const uint GROUP_SIZE = TF_GROUP_SIZE;

[[vk::binding(0,0)]] StructuredBuffer<uint> Params : register(t0, space0); // [0] = count, [1] = typeCode
[[vk::binding(1,0)]] StructuredBuffer<uint> Keys : register(t1, space0);   // original key bit patterns
[[vk::binding(2,0)]] RWStructuredBuffer<uint> Errors : register(u2, space0); // single uint atomic counter

// Map a key to sortable uint (same transform as map_to_uint.slang)
uint mapKey(uint raw, uint typeCode)
{
    if (typeCode == TYPE_INT)
    {
        raw ^= SIGN_BIT;
    }
    else if (typeCode == TYPE_FLOAT)
    {
        uint mask = ((raw >> 31) == 1u) ? FULL_MASK : SIGN_BIT;
        raw ^= mask;
    }
    return raw;
}

[numthreads(GROUP_SIZE, 1, 1)]
void csValidate(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    uint n = Params[0];
    if (n < 2 || i >= n - 1)
        return;

    uint typeCode = Params[1];

    uint a = mapKey(Keys[i], typeCode);
    uint b = mapKey(Keys[i + 1u], typeCode);

    if (a > b)
    {
        InterlockedAdd(Errors[0], 1u);
    }
}