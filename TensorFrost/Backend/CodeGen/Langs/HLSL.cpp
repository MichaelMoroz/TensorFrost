#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

string GenerateHLSLKernel(Program* program, const Kernel* kernel) {
	string final_source = R"(
uint pcg(uint v)
{
	uint state = v * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

float pcgf(uint v)
{
	return float(pcg(v)) / float(0xffffffffu)
}

RWStructuredBuffer<uint> mem : register(u0);

struct UBO
{
	uint var[32];
};

cbuffer ubo : register(b1) { UBO ubo; }

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
	int thread_id = dtid.x;
)";

	CodeGenerator generator;
	generator.GenerateKernelCode(kernel);
	string kernel_code = generator.AssembleString();

	final_source += AddIndent(kernel_code, "    ") + "\n}\n";

    return final_source;
	
}

}  // namespace TensorFrost
