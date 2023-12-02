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
		} else if (op->op_type_ == OpType::Store || op->op_type_ == OpType::Load ||
		           op->op_type_ == OpType::Scatter) {
			string address = "offsets[" + to_string(offsets[memory[0].from_->get()]) +
			                 "] + " + arguments[1];
			string memory_expression = "memory[" + address + "]";
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
			else if (op->op_type_ == OpType::Scatter)
			{
				string input_type_name = type_names[input_types[0]];
				expression += op->code_ + "((" + input_type_name + "*)memory, " +
				              address + ", " + arguments[2] + ")";
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

pair<string, vector<string>> GenerateC(Program* program) {
	string all_kernels = R"(
#include <cmath>
#include <omp.h>
typedef unsigned int uint;

inline float asfloat(uint x)
{
  return *(float*)&x;
}

inline uint asuint(float x)
{
  return *(uint*)&x;
}

inline int clamp(int x, int min, int max)
{
  if(x < min) return min;
  if(x > max) return max;
  return x;
}

inline float clamp(float x, float min, float max)
{
  return fmin(fmax(x, min), max);
}

inline double clamp(double x, double min, double max)
{
  return fmin(fmax(x, min), max);
}

inline void InterlockedAdd(int* memory, int address, int value)
{
  #pragma omp atomic
  memory[address] += value;
}

inline void InterlockedAdd(uint* memory, int address, uint value)
{
  #pragma omp atomic
  memory[address] += value;
}

inline void InterlockedAdd(float* memory, int address, float value)
{
  #pragma omp atomic
  memory[address] += value;
}

inline void InterlockedAnd(int* memory, int address, int value)
{
  #pragma omp atomic
  memory[address] &= value;
}

inline void InterlockedAnd(uint* memory, int address, uint value)
{
  #pragma omp atomic
  memory[address] &= value;
}

inline void InterlockedOr(int* memory, int address, int value)
{
  #pragma omp atomic
  memory[address] |= value;
}

inline void InterlockedOr(uint* memory, int address, uint value)
{
  #pragma omp atomic
  memory[address] |= value;
}

inline void InterlockedXor(int* memory, int address, int value)
{
  #pragma omp atomic
  memory[address] ^= value;
}

inline void InterlockedXor(uint* memory, int address, uint value)
{
  #pragma omp atomic
  memory[address] ^= value;
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
		kernel_names.push_back(kernel_name);

		// Generate kernel
		C_CodeGenerator generator;
		generator.GenerateKernelLines(program->ir_, cluster, kernel);
		generator.Compactify();

		string kernel_code = generator.GetFinalCode();
		kernel->generated_code_ = kernel_code;
		all_kernels +=
		    "\n"
		    "extern \"C\" __declspec(dllexport) void " + kernel_name +
		    "(uint* variables, uint* offsets, uint* memory, uint threads)\n"
		    "{\n"
		    "  #pragma omp parallel num_threads(64) \n"
			"  #pragma omp for \n"
		    "  for(int thread_id = 0; thread_id < threads; thread_id++)\n"
		    "  {\n" +
			AddIndent(kernel_code, "    ") +
		    "  }\n"
		    "}\n";
	}

	program->generated_code_ = all_kernels;
	return pair<string, vector<string>>(all_kernels, kernel_names);
}
}  // namespace TensorFrost