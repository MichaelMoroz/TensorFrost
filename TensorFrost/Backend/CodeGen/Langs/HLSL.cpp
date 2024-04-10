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
	                        const string& input, const string& output) override
	{
		if (op == "InterlockedAdd_Prev") {
			return "0; InterlockedAdd(mem[" + address + "], " + input + ", " + output + ")";
		}
		else
		{
			return op + "(mem[" + address + "], " + input + ")";
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
	return float(pcg(v)) / float(0xffffffffu)
}

RWStructuredBuffer<uint> mem : register(u0);

struct UBO
{
	int off[32];
	int var[32];
	int dispatch_size;
};

cbuffer ubo : register(b1) { UBO ubo; }

)";
}

void GenerateHLSLKernel(Program* program, Kernel* kernel) {
	string final_source = GetHLSLHeader();

	final_source += R"(

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
  int thread_id = dtid.x;
  
  if (thread_id >= ubo.dispatch_size) {
    return;
  }

)";

	HLSLGenerator generator;
	generator.GenerateKernelCode(kernel);
	string kernel_code = generator.AssembleString();

	final_source += AddIndent(kernel_code, "  ");

	final_source += "}\n";

    kernel->generated_code_ = final_source;
}

}  // namespace TensorFrost
