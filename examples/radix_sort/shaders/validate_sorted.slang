static const uint TYPE_UINT = 0u;
static const uint TYPE_INT = 1u;
static const uint TYPE_FLOAT = 2u;
static const uint SIGN_BIT = 0x80000000u;
static const uint FULL_MASK = 0xFFFFFFFFu;
static const uint GROUP_SIZE = TF_GROUP_SIZE;

struct ValidateParams {
    uint elementCount;
    uint typeCode;
};

[[vk::push_constant]]
ValidateParams gParams;

[[vk::binding(0,0)]] StructuredBuffer<uint> Keys : register(t0, space0);   // original key bit patterns
[[vk::binding(1,0)]] RWStructuredBuffer<uint> Errors : register(u1, space0); // single uint atomic counter

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
    uint n = gParams.elementCount;
    if (n < 2 || i >= n - 1)
        return;

    uint typeCode = gParams.typeCode;

    uint a = mapKey(Keys[i], typeCode);
    uint b = mapKey(Keys[i + 1u], typeCode);

    if (a > b)
    {
        InterlockedAdd(Errors[0], 1u);
    }
}
