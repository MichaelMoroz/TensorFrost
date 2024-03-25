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
		return "atomic" + op + "(" + address + ", " + input + ")";
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

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

uniform uint off[32];
uniform uint var[32];
uniform uint dispatch_size;

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
