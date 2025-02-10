#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

class HLSLGenerator : public CodeGenerator {
 public:
	HLSLGenerator(IR* ir) : CodeGenerator(ir) {
		name_map_ = {
			{"var", "var."},
			{"group_barrier", "GroupMemoryBarrierWithGroupSync"}
		};
	}

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
				return "InterlockedAddF("+memory_name+", " + address + ", " + input + ")";
			}
			return "InterlockedAdd("+memory_name+"[" + address + "], " + input + ")";
		} else if (op == "InterlockedAdd_Prev") {
			if(input_type_name == "float")
			{
				return "InterlockedAddF("+memory_name+", " + address + ", " + input + ")";
			}
			additional_lines.push_back("InterlockedAdd("+memory_name+"[" + address + "], " +
									   input + ", " + output + ");");
			return "0";
		} else {
			return op + "("+memory_name+"[" + address + "], " + input + ")";
		}
	}
};

string GetHLSLHeader(Kernel* kernel) {
	string header =R"(
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

)";
	kernel->var_names = vector<string>(kernel->variables.size());
	kernel->var_types = vector<string>(kernel->variables.size());
	header += "\nstruct UBO {\n";
	for (auto var : kernel->variables) {
		kernel->var_names[var.second] = var.first->var_name;
		kernel->var_types[var.second] = type_names[var.first->format.type];
	}
	kernel->var_names.push_back("_kernel_block_offset");
	kernel->var_types.push_back(type_names[TFType::Uint]);
	for (int i = 0; i < kernel->var_names.size(); i++) {
		header += "  " + kernel->var_types[i] + " " + kernel->var_names[i] + ";\n";
	}
	header += "};\n\n";
	return header;
}

string HLSLBufferDeclaration(const string& name, const string& type_name, const size_t binding) {
	return "RWStructuredBuffer<" + type_name + "> " + name + "_mem : register(u" + to_string(binding) + ");\n";
}

string HLSLGroupBufferDeclaration(const string& name, const string& type_name, const size_t size) {
	string decl = "groupshared " + type_name + " " + name + "[" + to_string(size) + "];\n";
	return decl;
}

void GenerateHLSLKernel(Program* program, Kernel* kernel) {
	kernel->generated_header_ = GetHLSLHeader(kernel);

	kernel->generated_bindings_ = GetBufferDeclarations(kernel, HLSLBufferDeclaration) + "\n";
	kernel->generated_bindings_ += "cbuffer ubo : register(b0) { UBO var; }\n";

	vector<int> group_size = kernel->root->group_size;
	// pad with 1s
	while (group_size.size() < 3) {
		group_size.push_back(1);
	}

	string main_function = "";

	main_function += GetGroupBufferDeclarations(kernel, HLSLGroupBufferDeclaration) + "\n";

	main_function += "[numthreads(" + to_string(group_size[0]) + ", " + to_string(group_size[1]) + ", " + to_string(group_size[2]) + ")]";

	main_function += R"(
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
  int block_id = gid.x + var._kernel_block_offset;
  int block_thread_id0 = gtid.x;
  int block_thread_id1 = gtid.y;
  int block_thread_id2 = gtid.z;

)";

	HLSLGenerator generator = HLSLGenerator(program->ir_);
	generator.GenerateKernelCode(kernel);
	string kernel_code = generator.AssembleString();

	main_function += AddIndent(kernel_code, "  ");

	main_function += "}\n";
	kernel->generated_main_ = main_function;

	kernel->full_generated_code_ = kernel->generated_header_ + kernel->generated_bindings_ + kernel->generated_main_;
}

}  // namespace TensorFrost