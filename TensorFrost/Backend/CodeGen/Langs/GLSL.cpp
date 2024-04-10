#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

class GLSLGenerator : public CodeGenerator {
	unordered_map<string, string> function_name_map_ = {
		{"modf", "mod"},
		{"atan2", "atan"},
		{"lerp", "mix"},
	};

 public:
	string TypeCast(string type_name, string input) override {
		return type_name + "(" + input + ")";
	}

	string GenerateAtomicOp(const string& op, const string& input_type_name,
	                        const string& output_type_name,
	                        const string& address, const string& input, const string& output) override {
		if (op == "InterlockedAdd") {
			return "atomicAdd(mem[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedAdd_Prev") {
			return output_type_name + "(atomicAdd(mem[" + address + "], uint(" + input +
			       ")))";
		} else if (op == "InterlockedMin") {
			return "atomicMin(mem[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedMax") {
			return "atomicMax(mem[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedAnd") {
			return "atomicAnd(mem[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedOr") {
			return "atomicOr(mem[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedXor") {
			return "atomicXor(mem[" + address + "], uint(" + input + "))";
		}
		else
		{
			throw runtime_error("Unsupported atomic operation: " + op);
		}
	}

	string GetFunctionName(const string& name) override {
		// Check if the function name is in the map
		if (function_name_map_.find(name) != function_name_map_.end()) {
			return function_name_map_[name];
		}

		// If not, return the original name
		return name;
	}
};

void GenerateGLSLKernel(Program* program, Kernel* kernel) {
	string final_source = R"(
#version 460

uint pcg(uint v) {
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

float pcgf(uint v) {
  return float(pcg(v)) / float(0xffffffffu);
}

float asfloat(uint x) {
  return uintBitsToFloat(x);
}

uint asuint(float x) {
  return floatBitsToUint(x);
}

uint asuint(int x) {
  return uint(x);
}

uint asuint(uint x) {
  return x;
}

int asint(uint x) {
  return int(x);
}

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

uniform int off[32];
uniform int var[32];
uniform int dispatch_size;

layout(std430, binding = 0) buffer memory {
  uint mem[];
};

void main() {
  int thread_id = int(gl_GlobalInvocationID.x);
  int block_id = int(gl_WorkGroupID.x);
  
  if (thread_id >= dispatch_size) {
    return;
  }

)";

	GLSLGenerator generator;
	generator.GenerateKernelCode(kernel);
	string kernel_code = generator.AssembleString();

	final_source += AddIndent(kernel_code, "  ");

	final_source += "}\n";

    kernel->generated_code_ = final_source;
}

}  // namespace TensorFrost
