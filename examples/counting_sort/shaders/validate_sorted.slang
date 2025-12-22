static const uint GROUP_SIZE = CS_GROUP_SIZE;

struct ValidateParams {
    uint elementCount;
};

[[vk::push_constant]]
ValidateParams gParams;

[[vk::binding(0,0)]] StructuredBuffer<uint> Keys : register(t0, space0);
[[vk::binding(1,0)]] RWStructuredBuffer<uint> Errors : register(u1, space0);

[numthreads(GROUP_SIZE, 1, 1)]
void csValidate(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    uint n = gParams.elementCount;
    if (n < 2 || index >= n - 1)
        return;

    uint a = Keys[index];
    uint b = Keys[index + 1u];
    if (a > b)
    {
        InterlockedAdd(Errors[0], 1u);
    }
}
