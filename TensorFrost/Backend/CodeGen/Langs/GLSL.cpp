#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

class GLSLGenerator : public CodeGenerator {
 public:
	GLSLGenerator(IR* ir) : CodeGenerator(ir) {
		name_map_ = {
			{"var", "var."},
			{"modf", "mod"},
			{"atan2", "atan"},
			{"lerp", "mix"},
			{"reversebits", "bitfieldReverse"},
			{"frac", "fract"},
			{"group_barrier", "barrier"}
		};
	}

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
			return "atomicAdd("+memory_name+"[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedAdd_Prev") {
			if(input_type_name == "float")
			{
				return  output_type_name + "(atomicAdd_"+memory_name+"(" + address + ", " + input +"))";
			}
			return output_type_name + "(atomicAdd("+memory_name+"[" + address + "], uint(" + input + ")))";
		} else if (op == "InterlockedMin") {
			return "atomicMin("+memory_name+"[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedMax") {
			return "atomicMax("+memory_name+"[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedAnd") {
			return "atomicAnd("+memory_name+"[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedOr") {
			return "atomicOr("+memory_name+"[" + address + "], uint(" + input + "))";
		} else if (op == "InterlockedXor") {
			return "atomicXor("+memory_name+"[" + address + "], uint(" + input + "))";
		}
		else
		{
			throw runtime_error("Unsupported atomic operation: " + op);
		}
	}

};

string GetGLSLHeader(Kernel* kernel) {
	string header = R"(
#version 430

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

uint asuint(bool x) {
	return uint(x);
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

bool asbool(uint x) {
  return bool(x);
}

)";
	kernel->var_names = vector<string>(kernel->variables.size());
	kernel->var_types = vector<string>(kernel->variables.size());
	header += "\nstruct UBO {\n";
	for (auto var : kernel->variables) {
		kernel->var_names[var.second] = var.first->var_name;
		kernel->var_types[var.second] = type_names[var.first->type];
	}
	for (int i = 0; i < kernel->var_names.size(); i++) {
		header += "  " + kernel->var_types[i] + " " + kernel->var_names[i] + ";\n";
	}
	if(kernel->var_names.size() == 0)
	{
		header += "  uint dummy;\n";
	}
	header += "};\n\n";
	return header;
}

string GLSLBufferDeclaration(const string& name, const string& type_name, const size_t binding) {
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

string GLSLGroupBufferDeclaration(const string& name, const string& type_name, const size_t size) {
	string decl = "shared " + type_name + " " + name + "[" + to_string(size) + "];\n";
	return decl;
}

void GenerateGLSLKernel(Program* program, Kernel* kernel) {
	kernel->generated_header_ = GetGLSLHeader(kernel);

	string buffers = GetBufferDeclarations(kernel, GLSLBufferDeclaration) + "\n";
	buffers += "layout(std140) uniform UBOBlock {\n  UBO var;\n};\n\n";
	kernel->generated_bindings_ = buffers;

	string main_code = "";

	main_code += GetGroupBufferDeclarations(kernel, GLSLGroupBufferDeclaration) + "\n";

	vector<int> group_size = kernel->root->group_size;
	//pad with 1s
	while (group_size.size() < 3) {
		group_size.push_back(1);
	}

	main_code += "layout (local_size_x = " + to_string(group_size[0]) + ", local_size_y = " + to_string(group_size[1]) + ", local_size_z = " + to_string(group_size[2]) + ") in;\n";


	main_code += R"(
void main() {
  int block_id = int(gl_WorkGroupID.x);
  int block_thread_id0 = int(gl_LocalInvocationID.x);
  int block_thread_id1 = int(gl_LocalInvocationID.y);
  int block_thread_id2 = int(gl_LocalInvocationID.z);

)";

	GLSLGenerator generator = GLSLGenerator(program->ir_);
	generator.GenerateKernelCode(kernel);
	string kernel_code = generator.AssembleString();

	main_code += AddIndent(kernel_code, "  ");

	main_code += "}\n";

    kernel->generated_main_ = main_code;

	kernel->full_generated_code_ = kernel->generated_header_ + kernel->generated_bindings_ + kernel->generated_main_;
}

}  // namespace TensorFrost