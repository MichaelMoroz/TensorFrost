#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

class C_CodeGenerator : public CodeGenerator {
 public:
	Line* GenerateLine(NodeNames* names, const Operation* op, Node* node,
	                   Arguments inputs, Arguments indices, Arguments shape,
	                   Arguments memory, map<Node*, int> offsets,
	                   map<Node*, int> variables) override {
		// get node names
		vector<string> arguments;
		vector<string> input_variables;
		vector<DataType> input_types;
		for (const Arg& arg : memory) {
			string var_name = GetNodeName(arg.from_->get(), *names, true);
			arguments.push_back(var_name);
			if (arg.from_->get()->name != "const" &&
			    arg.from_->get()->name != "memory") {
				input_variables.push_back(var_name);
			}
		}
		for (const Arg& arg : indices) {
			string var_name = GetNodeName(arg.from_->get(), *names, true);
			arguments.push_back(var_name);
			if (arg.from_->get()->name != "const" &&
			    arg.from_->get()->name != "memory") {
				input_variables.push_back(var_name);
			}
		}
		for (const Arg& arg : inputs) {
			Node* input = arg.from_->get();
			string name = GetNodeName(input, *names, true);
			if (input->name == "memory") {
				name = "variables[" + to_string(variables[input]) + "]";
			}
			arguments.push_back(name);
			input_types.push_back(arg.from_->get()->tensor_->type);
			if (input->name != "const" && input->name != "memory") {
				input_variables.push_back(name);
			}
		}

		string name = (*names)[node];

		// get output type
		DataType output_type = op->GetOutputType(input_types);

		map<DataType, string> type_names = {
		    {DataType::None, "void"},   {DataType::Bool, "bool"},
		    {DataType::Float, "float"}, {DataType::Uint, "uint"},
		    {DataType::Int, "int"},
		};

		// generate line
		string left = "";
		string expression = "";
		string right = "";
		bool needs_parenthesis = true;
		if (op->name_ == "loop_begin") {
			left += "for (int " + name + " = " + arguments[0] + "; " + name + " < " +
			        arguments[1] + "; " + name + " += " + arguments[2] + ") {";
		} else if (op->name_ == "loop_end") {
			left += "}";
		} else if (op->name_ == "if_begin") {
			left += "if (" + arguments[0] + ") {";
		} else if (op->name_ == "if_end") {
			left += "}";
		} else if (op->op_type_ == OpType::Store || op->op_type_ == OpType::Load) {
			string memory_expression = "memory[offsets[" +
			                           to_string(offsets[memory[0].from_->get()]) +
			                           "] + " + arguments[1] + "]";
			if (op->name_ == "load") {
				left += type_names[output_type] + " " + name + " = ";
				if (output_type == DataType::Float) {
					expression += "asfloat(";
				}
				if (output_type == DataType::Int) {
					expression += "asint(";
				}
				expression += memory_expression;
				if (output_type != DataType::Uint) {
					expression += ")";
				}
				right += ";";
				needs_parenthesis = false;
			} else if (op->name_ == "store") {
				expression += memory_expression + " = ";
				if (input_types[0] != DataType::Uint) {
					expression += "asuint(";
				}
				expression += arguments[2];
				if (input_types[0] != DataType::Uint) {
					expression += ")";
				}
				right += ";";
			}
		} else {
			if (output_type != DataType::None) {
				left += type_names[output_type] + " " + name + " = ";
			}
			string line;

			switch (op->op_type_) {
				case OpType::Operator:
					line += arguments[0] + " " + op->code_ + " " + arguments[1];
					break;
				case OpType::UnaryOperator:
					line += op->code_ + arguments[0];
					break;
				case OpType::Function:
					line += op->code_ + "(";
					for (int i = 0; i < arguments.size(); i++) {
						if (i != 0) {
							line += ", ";
						}
						line += arguments[i];
					}
					line += ")";
					needs_parenthesis = false;
					break;
				case OpType::Keyword:
					line += op->code_;
					break;
				case OpType::Variable:
					line += op->code_;
					needs_parenthesis = false;
					break;
				case OpType::TypeCast:
					line += "(" + op->code_ + ")" + arguments[0];
					break;
				case OpType::TypeReinterpret:
					line += "*(" + op->code_ + "*)&" + arguments[0];
					break;
				case OpType::Constant:
					line += node->tensor_->GetConstantString();
					needs_parenthesis = false;
					break;
				default:
					line += "";
					break;
			}
			expression += line;
			right += ";";
		}

		return new Line(left, expression, right, name, input_variables,
		                needs_parenthesis, op->cost_);
	}
};

string GenerateKernelC(const IR* ir, const Lable* cluster,
                       const Kernel* kernel,
                       const string& kernel_name) {
	//NodeNames names = GenerateNodeNames(*ir);

	string hlsl_code;

	// Begin HLSL function
	hlsl_code += "void " + kernel_name + "(";
	hlsl_code += "uint* variables, ";
	hlsl_code += "uint* offsets, ";
	hlsl_code += "uint* memory, ";
	hlsl_code += "uint thread_id";
	hlsl_code += ")\n";
	hlsl_code += "{\n";

	C_CodeGenerator generator;
	generator.GenerateKernelLines(ir, cluster, kernel);
	generator.Compactify();

	hlsl_code += generator.GetFinalCode();

	// End HLSL function
	hlsl_code += "}\n";

	return hlsl_code;
}

pair<string, vector<string>> GenerateC(Program* program) {
	string all_kernels = R"(
#include <cmath>
#include <omp.h>
typedef unsigned int uint;

float asfloat(uint x)
{
  return *(float*)&x;
}

uint asuint(float x)
{
  return *(uint*)&x;
}

int clamp(int x, int min, int max)
{
  if(x < min) return min;
  if(x > max) return max;
  return x;
}

float clamp(float x, float min, float max)
{
  return fmin(fmax(x, min), max);
}

double clamp(double x, double min, double max)
{
  return fmin(fmax(x, min), max);
}

void atomic_add(int* address, int value)
{
  #pragma omp atomic
  *address += value;
}

void atomic_add(uint* address, uint value)
{
  #pragma omp atomic
  *address += value;
}

void atomic_add(float* address, float value)
{
  #pragma omp atomic
  *address += value;
}

void atomic_and(int* address, int value)
{
  #pragma omp atomic
  *address &= value;
}

void atomic_and(uint* address, uint value)
{
  #pragma omp atomic
  *address &= value;
}

void atomic_or(int* address, int value)
{
  #pragma omp atomic
  *address |= value;
}

void atomic_or(uint* address, uint value)
{
  #pragma omp atomic
  *address |= value;
}

void atomic_xor(int* address, int value)
{
  #pragma omp atomic
  *address ^= value;
}

void atomic_xor(uint* address, uint value)
{
  #pragma omp atomic
  *address ^= value;
}

)";

	// Generate HLSL code for each cluster
	int kernel_count = 0;
	vector<string> kernel_names;
	for (auto& i : program->kernels_) {
		Kernel* kernel = &i;
		Lable* cluster = kernel->begin_->cluster_head_;
		if (kernel->type_ != KernelType::Compute) {
			continue;
		}

		string kernel_name = "kernel_" + to_string(kernel_count++);
		string function_name = kernel_name + "_execute";
		kernel_names.push_back(function_name);

		// Generate kernel
		all_kernels += "\n";
		string kernel_code = GenerateKernelC(program->ir_, cluster, kernel, kernel_name);
		kernel->generate_code_ = kernel_code;
		all_kernels += kernel_code;
		all_kernels +=
		    "\n"
		    "extern \"C\" \n"
		    "{ \n"
		    "  __declspec(dllexport) void " + function_name +
		    "(uint* variables, uint* offsets, uint* memory, uint threads)\n"
		    "  {\n"
		    "    #pragma omp parallel for\n"
		    "    for(int i = 0; i < threads; i++)\n"
		    "    {\n"
		    "      " + kernel_name + "(variables, offsets, memory, i);\n"
		    "    }\n"
		    "  }\n"
		    "}\n";
	}

	program->generate_code_ = all_kernels;
	return pair<string, vector<string>>(all_kernels, kernel_names);
}
}  // namespace TensorFrost