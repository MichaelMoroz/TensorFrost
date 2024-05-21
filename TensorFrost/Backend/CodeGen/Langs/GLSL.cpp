#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

class GLSLGenerator : public CodeGenerator {
	unordered_map<string, string> name_map_ = {
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
	                        const string& address, const string& input, const string& output, const string& memory_name) override {
		if (op == "InterlockedAdd") {
			if(input_type_name == "float")
			{
				return "atomicAdd_"+memory_name+"(" + address + ", " + input + ")";
			}
			return "atomicAdd("+memory_name+"_mem[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedAdd_Prev") {
			if(input_type_name == "float")
			{
				return  output_type_name + "(atomicAdd_"+memory_name+"(" + address + ", " + input +"))";
			}
			return output_type_name + "(atomicAdd("+memory_name+"_mem[" + address + "], uint(" + input + ")))";
		} else if (op == "InterlockedMin") {
			return "atomicMin("+memory_name+"_mem[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedMax") {
			return "atomicMax("+memory_name+"_mem[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedAnd") {
			return "atomicAnd("+memory_name+"_mem[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedOr") {
			return "atomicOr("+memory_name+"_mem[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedXor") {
			return "atomicXor("+memory_name+"_mem[" + address + "], uint(" + input + "))";
		}
		else
		{
			throw runtime_error("Unsupported atomic operation: " + op);
		}
	}

	string GetName(const string& name) override {
		// Check if the function name is in the map
		if (name_map_.find(name) != name_map_.end()) {
			return name_map_[name];
		}

		// If not, return the original name
		return name;
	}
};

string GetGLSLHeader()
{
	return R"(
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

uniform int var[32];
)";
}

string GLSLBufferDeclaration(const string& name, const string& type_name, const int binding) {
	string decl = "layout(std430, binding = " + to_string(binding) + ") buffer buf_" + name + " {\n  " + type_name + " " + name + "_mem[];\n};\n";
	//add atomic functions
	decl += R"(
float atomicAdd_)" + name + R"((int index, float val) {
	uint uval = floatBitsToUint(val);
	uint tmp0 = 0;
	uint tmp1 = 0;

	while (true) {
		tmp0 = atomicCompSwap()" + name + R"(_mem[index], tmp1, uval);
		if (tmp1 == tmp0) break;
		tmp1 = tmp0;
		uval = floatBitsToUint(val + uintBitsToFloat(tmp1));
	}

	return uintBitsToFloat(tmp1);
}

)";

	return decl;
}

void GenerateGLSLKernel(Program* program, Kernel* kernel) {
	string final_source = GetGLSLHeader();

	//add buffer declarations
	for (auto& buffer : kernel->memory) {
		Node* mem_node = buffer.first;
		int binding = buffer.second;
		string name = mem_node->var_name;
		string type_name = "uint";
		final_source += GLSLBufferDeclaration(name, type_name, binding);
	}

	vector<int> group_size = kernel->root->group_size;
	//reverse vector
	reverse(group_size.begin(), group_size.end());
	//pad with 1s
	while (group_size.size() < 3) {
		group_size.push_back(1);
	}

	final_source += "layout (local_size_x = " + to_string(group_size[0]) + ", local_size_y = " + to_string(group_size[1]) + ", local_size_z = " + to_string(group_size[2]) + ") in;\n";


	final_source += R"(
void main() {
  int block_id = int(gl_WorkGroupID.x);
  int block_thread_id0 = int(gl_LocalInvocationID.x);
  int block_thread_id1 = int(gl_LocalInvocationID.y);
  int block_thread_id2 = int(gl_LocalInvocationID.z);

)";

	GLSLGenerator generator;
	generator.GenerateKernelCode(kernel);
	string kernel_code = generator.AssembleString();

	final_source += AddIndent(kernel_code, "  ");

	final_source += "}\n";

    kernel->generated_code_ = final_source;
}

}  // namespace TensorFrost
