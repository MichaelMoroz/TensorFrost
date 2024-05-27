#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

class HLSLGenerator : public CodeGenerator {
	unordered_map<string, string> name_map_ = {
	    {"off", "ubo.off"},
	    {"var", "ubo.var"},
	    {"dispatch_size", "ubo.dispatch_size"},

	};

 public:
	string TypeCast(string type_name, string input) override {
		return type_name + "(" + input + ")";
	}

	string GenerateAtomicOp(const string& op, const string& input_type_name,
	                        const string& output_type_name, const string& address,
	                        const string& input, const string& output, const string& memory_name) override
	{
		if (op == "InterlockedAdd") {
			if(input_type_name == "float")
			{
				return "InterlockedAddF("+memory_name+"_mem, " + address + ", " + input + ")";
			}
			return "InterlockedAdd("+memory_name+"_mem[" + address + "], " + input + ")";
		} else if (op == "InterlockedAdd_Prev") {
			if(input_type_name == "float")
			{
				return "InterlockedAddF("+memory_name+"_mem, " + address + ", " + input + ")";
			}
			additional_lines.push_back("InterlockedAdd("+memory_name+"_mem[" + address + "], " +
									   input + ", " + output + ");");
			return "0";
		} else {
			return op + "("+memory_name+"_mem[" + address + "], " + input + ")";
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

string GetHLSLHeader() { 
	return R"(
uint pcg(uint v)
{
	uint state = v * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

float pcgf(uint v)
{
	return float(pcg(v)) / float(0xffffffffu);
}

float InterlockedAddF(RWStructuredBuffer<uint> buffer, int index, float val)
{
    uint uval = asuint(val), tmp0 = 0, tmp1 = 0;
    [allow_uav_condition] while (true) {
        InterlockedCompareExchange(buffer[index], tmp0, uval, tmp1);
        if (tmp1 == tmp0)  break;
        tmp0 = tmp1;
        uval = asuint(val + asfloat(tmp1));
    }
    return asfloat(tmp1);
}

struct UBO
{
	int var[32];
};

cbuffer ubo : register(b0) { UBO ubo; }

)";
}

string HLSLBufferDeclaration(const string& name, const string& type_name, const int binding) {
	return "RWStructuredBuffer<" + type_name + "> " + name + "_mem : register(u" + to_string(binding) + ");\n";
}

void GenerateHLSLKernel(Program* program, Kernel* kernel) {
	string final_source = GetHLSLHeader();

	final_source += GetBufferDeclarations(kernel, HLSLBufferDeclaration);
	final_source += "\n";

	vector<int> group_size = kernel->root->group_size;
	// reverse vector
	reverse(group_size.begin(), group_size.end());
	// pad with 1s
	while (group_size.size() < 3) {
		group_size.push_back(1);
	}

	final_source += "[numthreads(" + to_string(group_size[0]) + ", " + to_string(group_size[1]) + ", " + to_string(group_size[2]) + ")]";

	final_source += R"(
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
  int block_id = gid.x;
  int block_thread_id0 = gtid.x;
  int block_thread_id1 = gtid.y;
  int block_thread_id2 = gtid.z;

)";

	HLSLGenerator generator;
	generator.GenerateKernelCode(kernel);
	string kernel_code = generator.AssembleString();

	final_source += AddIndent(kernel_code, "  ");

	final_source += "}\n";

    kernel->generated_code_ = final_source;
}

}  // namespace TensorFrost
