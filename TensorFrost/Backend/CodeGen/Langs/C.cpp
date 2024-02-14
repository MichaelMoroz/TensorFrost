#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

class C_CodeGenerator : public CodeGenerator {
 public:
	Line* GenerateLine(const Operation* op, Node* node,
	                   Arguments inputs, Arguments indices, Arguments shape,
	                   Arguments memory, map<Node*, int> offsets,
	                   map<Node*, int> variables) override {
		// get node names
		vector<string> arguments;
		vector<string> input_variables;
		vector<DataType> input_types;
		for (const Arg& arg : memory) {
			string var_name = GetNodeName(arg.from_->get(), true);
			arguments.push_back(var_name);
			if (arg.from_->get()->name != "const" &&
			    arg.from_->get()->name != "memory") {
				input_variables.push_back(var_name);
			}
		}
		for (const Arg& arg : indices) {
			string var_name = GetNodeName(arg.from_->get(), true);
			arguments.push_back(var_name);
			if (arg.from_->get()->name != "const" &&
			    arg.from_->get()->name != "memory") {
				input_variables.push_back(var_name);
			}
		}
		for (const Arg& arg : inputs) {
			Node* input = arg.from_->get();
			string name = GetNodeName(input, true);
			if (input->name == "memory") {
				name = "var[" + to_string(variables[input]) + "]";
			}
			arguments.push_back(name);
			input_types.push_back(arg.from_->get()->GetTensor()->type);
			if (input->name != "const" && input->name != "memory") {
				input_variables.push_back(name);
			}
		}

		vector<string> shapes;
		for (const Arg& arg : shape) {
			string name = GetNodeName(arg.from_->get(), true);
			shapes.push_back(name);
		}

		string name = node->var_name;

		// get output type
		DataType output_type = node->tensor_->type;//op->GetOutputType(input_types);

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
		} else if (op->HasAllTypes(OpType::MemoryOp)) {
			string address = "off[" + to_string(offsets[memory[0].from_->get()]) +
			                 "] + " + arguments[1];
			string memory_expression = "mem[" + address + "]";
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
			else if (op->HasAllTypes(OpType::Scatter))
			{
				string input_type_name = type_names[input_types[0]];
				expression += op->code_ + "((" + input_type_name + "*)mem, " +
				              address + ", " + arguments[2] + ")";
				right += ";";
			}
		} else if (op->name_ == "set") {
			left += arguments[0] + " = ";
			expression += arguments[1];
			right += ";";
		} else {
			if (output_type != DataType::None) {
				left += type_names[output_type] + " " + name + " = ";
			}
			string line;

			switch (op->op_types_[0]) { //TODO: properly support multiple op types
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
				case OpType::DimensionIndex:
					line += op->code_ + to_string(node->GetTensor()->data[0]);
					needs_parenthesis = false;
					break;
				case OpType::TypeCast:
					line += "(" + op->code_ + ")" + arguments[0];
					break;
				case OpType::TypeReinterpret:
					line += "*(" + op->code_ + "*)&" + arguments[0];
					break;
				case OpType::Constant:
					line += node->GetTensor()->GetConstantString();
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

string ReadVariable(Node* node) {
	if (node->name == "const") {
		return to_string(node->GetTensor()->data[0]);
	} else {
		return "mem[" + node->var_name + "]";
	}
}

pair<string, vector<string>> GenerateC(Program* program) {
	string all_kernels = R"(
#include <cmath>
#include <omp.h>
#include <initializer_list>
#include <functional>
#include <vector>

typedef unsigned int uint;

inline int min(int a, int b)
{
  return a < b ? a : b;
}

inline int max(int a, int b)
{
  return a > b ? a : b;
}

inline float min(float a, float b)
{
  return a < b ? a : b;
}

inline float max(float a, float b)
{
  return a > b ? a : b;
}

inline float asfloat(uint x)
{
  return *(float*)&x;
}

inline uint asuint(float x)
{
  return *(uint*)&x;
}

inline int asint(uint x)
{
  return *(int*)&x;
}

inline uint asuint(int x)
{
  return *(uint*)&x;
}

inline int clamp(int x, int a, int b)
{
  return min(max(x, a), b);
}

inline float clamp(float x, float a, float b)
{
  return min(max(x, a), b);
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

inline void InterlockedMin(int* memory, int address, int value)
{
  #pragma omp critical
  memory[address] = min(memory[address], value);
}

inline void InterlockedMin(float* memory, int address, float value)
{
  #pragma omp critical
  memory[address] = min(memory[address], value);
}

inline void InterlockedMax(int* memory, int address, int value)
{
  #pragma omp critical
  memory[address] = max(memory[address], value);
}

inline void InterlockedMax(float* memory, int address, float value)
{
  #pragma omp critical
  memory[address] = max(memory[address], value);
}

inline uint pcg(uint v)
{
	uint state = v * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

inline float pcgf(uint v)
{
	return (float)pcg(v) / 4294967296.0f;
}

void dispatch(void(*kernel)(uint*, uint*, uint*, uint*), uint* mem, std::initializer_list<uint> off, std::initializer_list<uint> var, std::initializer_list<uint> shape)
{
  uint* off_arr = new uint[off.size()];
  uint* var_arr = new uint[var.size()];
  uint* shape_arr = new uint[shape.size()];

  for (int i = 0; i < off.size(); i++)
  {
    off_arr[i] = off.begin()[i];
  }

  for (int i = 0; i < var.size(); i++)
  {
	var_arr[i] = var.begin()[i];
  }

  for (int i = 0; i < shape.size(); i++)
  {
	shape_arr[i] = shape.begin()[i];
  }

  kernel(var_arr, off_arr, mem, shape_arr);

  delete[] off_arr;
  delete[] var_arr;
  delete[] shape_arr;
} 

)";
	
	GenerateNodeNames(*program->ir_);
	vector<string> allocated_memories;
	vector<Node*> output_memories;
	int input_memory_index = 0;

	// Generate HLSL code for each compute kernel
	int kernel_count = 0;
	vector<string> kernel_names;
	string host_code =
	    "\n"
	    "extern \"C\" "
#ifdef _WIN32
	    "__declspec(dllexport)"
#endif
	    " void "
	    "main"
	    "(uint* in, uint* out, uint* mem, uint allocate(uint*&, uint*, uint dim), "
	    "void deallocate(uint))\n"
	    "{\n";

	for (auto& i : program->kernels_) {
		Kernel* kernel = &i;
		Scope* cluster = kernel->begin_->kernel_;

		if (kernel->type_ == KernelType::Memory) {

			string memory_code = "";
			for (auto node = IR::Iterator(cluster->begin_);
			     !node.is_cluster_end(cluster); ++node) {
				if (node->name == "memory") {
					string left = "uint " + node->var_name + " = ";
					//if input memory type then just take the input and store it in the output
					if (node->memory_type_ == MemoryType::Input || node->memory_type_ == MemoryType::Shape) {
						memory_code += left + "in[" + to_string(input_memory_index++) + "];\n";
					}
					//if any other memory type - allocate it
					else {
						// get shape arguments
						ArgMap args = node->GetArgumentMap(Arg::Shape);
						uint dims = args.size();
						string shape_name = "shape_" + node->var_name;
						memory_code += "std::vector<uint> " + shape_name + " = {";
						if (dims == 0) {
							memory_code += "1";
						} else {
							for (int j = 0; j < dims; j++) {
								if (j != 0) {
									memory_code += ", ";
								}
								Node* shape_node = args[j]->from_->get();
						
								memory_code += ReadVariable(shape_node);
							}
						}

						memory_code += "};\n";

						memory_code += left + "allocate(mem, " + shape_name + ".data(), " + to_string(dims) + ");\n";

						if (node->memory_type_ == MemoryType::Output)
						{
							output_memories.push_back(*node);
						}
						else
						{
							allocated_memories.push_back(node->var_name);
						}
					}
				}
				else
				{
					throw std::runtime_error("Invalid kernel");
				}
			}

			host_code += AddIndent(memory_code, "  ");
			host_code += "\n";

			continue;
		}

		string kernel_name = "kernel_" + to_string(kernel_count++);
		kernel_names.push_back(kernel_name);

		// Generate kernel
		C_CodeGenerator generator;
		generator.GenerateKernelLines(program->ir_, cluster, kernel);
		//generator.Compactify();

		
		string kernel_code = generator.GetFinalCode();
		kernel->generated_code_ = kernel_code;

		if (i.dim == 0) //add it to host code if scalar
		{
			host_code += AddIndent(kernel_code, "  ");
		}
		else
		{
			string loop = "";
			const int block_size = 4;  // TODO chose automatically
			switch (kernel->indexing_mode_) {
				case KernelIndexingMode::Linear:
					loop =
					    "  for (int thread_id = 0; thread_id < shape[0]; thread_id++)\n";
					loop += "  {\n";
					break;
				case KernelIndexingMode::MultiDimensional:
					for (int d = 0; d < i.dim; d++) {
						loop += "  for (int dim" + to_string(d) + " = 0; dim" +
						        to_string(d) + " < shape[" + to_string(d) + "]; dim" +
						        to_string(d) + "++)\n";
					}
					loop += "  {\n";
					break;
				case KernelIndexingMode::MultiDimensionalBlocks:
					for (int d = 0; d < i.dim; d++) {
						loop += "  for (int wg" + to_string(d) + " = 0; wg" + to_string(d) +
						        " < shape[" + to_string(d) + "]; wg" + to_string(d) +
						        "+= " + to_string(block_size) + ")\n";
					}
					for (int d = 0; d < i.dim; d++) {
						loop += "  for (int dim" + to_string(d) + " = wg" + to_string(d) +
						        "; dim" + to_string(d) + " < min(wg" + to_string(d) + "+" +
						        to_string(block_size) + ", shape[" + to_string(d) +
						        "]); dim" + to_string(d) + "++)\n";
					}
					loop += "  {\n";
					break;
				default:
					throw std::runtime_error("Invalid indexing mode");
					break;
			}


			all_kernels +=
			    "\n"
			    "extern \"C\" "
#ifdef _WIN32
			    "__declspec(dllexport)"
#endif
			    " void " +
			    kernel_name +
			    "(uint* var, uint* off, uint* mem, uint* shape)\n"
			    "{\n"
			    "  #pragma omp parallel for shared(mem) \n" +
			    loop + AddIndent(kernel_code, "    ") +
			    "  }\n"
			    "}\n";

			host_code += "  dispatch(" + kernel_name + ", mem, {";

			vector<Node*> memory_nodes;
			memory_nodes.resize(kernel->memory.size());
			for (auto& memory : kernel->memory) {
				memory_nodes[memory.second] = memory.first;
			}

			vector<Node*> variable_nodes;
			variable_nodes.resize(kernel->variables.size());
			for (auto& variable : kernel->variables) {
				variable_nodes[variable.second] = variable.first;
			}

			for (int d = 0; d < memory_nodes.size(); d++) {
				if (d != 0) {
					host_code += ", ";
				}
				host_code += memory_nodes[d]->var_name;
			}
			host_code += "}, {";
			for (int d = 0; d < variable_nodes.size(); d++) {
				if (d != 0) {
					host_code += ", ";
				}
				host_code += ReadVariable(variable_nodes[d]);
			}
			host_code += "}, {";
			for (int d = 0; d < i.dim; d++) {
				if (d != 0) {
					host_code += ", ";
				}
				host_code += ReadVariable(i.shape[d]->from_->get());
			}
			host_code += "});\n";
			host_code += "\n";
		}
	}

	//set output memories and deallocate
	for (auto& memory : output_memories) {
		int output_memory_index = memory->memory_index_;
		string mem_name = memory->var_name;
		host_code +=
		    "  out[" + to_string(output_memory_index++) + "] = " + mem_name + ";\n";
	}

	//TODO: deallocate exactly after the last use for better memory management
	for (auto& memory : allocated_memories) {
		host_code += "  deallocate(" + memory + ");\n";
	}

	all_kernels += host_code + "}\n";

	program->generated_code_ = all_kernels;
	return pair<string, vector<string>>(all_kernels, kernel_names);
}
}  // namespace TensorFrost