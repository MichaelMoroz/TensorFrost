#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

class GLSLGenerator : public CodeGenerator {
 public:
	string TypeCast(string type_name, string input) override {
		return type_name + "(" + input + ")";
	}

	string GenerateAtomicOp(const string& op, const string& input_type_name, const string& address, const string& input)
	{
		if (op == "InterlockedAdd") {
			return "atomicAdd(mem[" + address + "], " + input + ")";
		}
		else if (op == "InterlockedMin") {
			return "atomicMin(mem[" + address + "], " + input + ")";
		}
		else if (op == "InterlockedMax") {
			return "atomicMax(mem[" + address + "], " + input + ")";
		}
		else if (op == "InterlockedAnd") {
			return "atomicAnd(mem[" + address + "], " + input + ")";
		}
		else if (op == "InterlockedOr") {
			return "atomicOr(mem[" + address + "], " + input + ")";
		}
		else if (op == "InterlockedXor") {
			return "atomicXor(mem[" + address + "], " + input + ")";
		}
		else {
			throw runtime_error("Unsupported atomic operation: " + op);
		}
	}
};

void GenerateGLSLKernel(Program* program, Kernel* kernel) {
	string final_source = R"(
#version 430 core

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

float asfloat(uint x)
{
  return uintBitsToFloat(x);
}

uint asuint(float x)
{
  return floatBitsToUint(x);
}

uint asuint(int x)
{
  return intBitsToUint(x);
}

uint asuint(uint x)
{
  return x;
}

int asint(uint x)
{
  return intBitsToUint(x);
}

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

uniform int off[32];
uniform int var[32];
uniform int dispatch_size;

layout(std430, binding = 0) buffer memory
{
	uint mem[];
};

void main()
{
	uint thread_id = gl_GlobalInvocationID.x;
	uint block_id = gl_WorkGroupID.x;

	if (thread_id >= dispatch_size)
	{
		return;
	}

)";

	GLSLGenerator generator;
	generator.GenerateKernelCode(kernel);
	string kernel_code = generator.AssembleString();

	final_source += AddIndent(kernel_code, "    ");

	final_source += "}\n";

    kernel->generated_code_ = final_source;
}

}  // namespace TensorFrost
