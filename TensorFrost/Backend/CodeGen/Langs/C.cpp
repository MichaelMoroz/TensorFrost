#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

string ReadVariable(Node* node) {
	if (node->name == "const") {
		return to_string(node->GetTensor()->data[0]);
	}
	if (node->name == "memory") {
		return "mem[" + node->var_name + "]";
	}
	return node->var_name;
}

class C_CodeGenerator : public CodeGenerator {
 public:
	string offset_name_ = "off";
	string variable_name_ = "var";
	string memory_name_ = "mem";

	int* input_memory_index;
	vector<string>* allocated_memories;
	vector<Node*>* output_memories;

	bool offset_array = true;

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
				name = variable_name_ + "[" + to_string(variables[input]) + "]";
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
			string address;
			if (offset_array)
			{
				address = offset_name_ + "[" + to_string(offsets[memory[0].from_->get()]) + "]";
			}
			else
			{
				address = arguments[0];
			}
			    
			if (arguments.size() > 1) { //if has index (not a scalar)
				address += " + " + arguments[1];
			}
			string memory_expression = memory_name_ + "[" + address + "]";
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
				if (output_type != DataType::None) {
					left += type_names[output_type] + " " + name + " = ";
				}
				string input_type_name = type_names[input_types[0]];
				expression += op->code_ + "((" + input_type_name + "*)" + memory_name_ +
				              ", " + address + ", " + arguments[2] + ")";
				right += ";";
			}
		} else if (op->name_ == "set") {
			left += arguments[0] + " = ";
			expression += arguments[1];
			right += ";";
		} else if (op->name_ == "memory") {
			left += "uint " + node->var_name + " = ";
			// if input memory type then just take the input and store it in the
			// output
			if (node->memory_type_ == MemoryType::Input ||
				node->memory_type_ == MemoryType::Shape) {
				expression += "in[" + to_string((*input_memory_index)++) + "]";
				right += ";";
			}
			// if any other memory type - allocate it
			else {
				// get shape arguments
				ArgMap args = node->GetArgumentMap(Arg::Shape);
				uint dims = args.size();
			
				string shape_arg = "{";
				if (dims == 0) {
					shape_arg += "1";
				} else {
					for (int j = 0; j < dims; j++) {
						if (j != 0) {
							shape_arg += ", ";
						}
						Node* shape_node = args[j]->from_->get();

						shape_arg += "(uint)" + ReadVariable(shape_node);
					}
				}

				shape_arg += "}";

				expression += "allocate(alloc, mem, " + shape_arg + ")";
				right += ";";

				if (node->memory_type_ == MemoryType::Output) {
					output_memories->push_back(node);
				}
				else {
					allocated_memories->push_back(node->var_name);
				}
			}
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

pair<string, vector<string>> GenerateC(Program* program) {
	string all_kernels = R"(
#include <cmath>
#include <omp.h>
#include <initializer_list>
#include <functional>
#include <vector>
#include <atomic>

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
  std::atomic<int>* place = reinterpret_cast<std::atomic<int>*>(&memory[address]);
  place->fetch_add(value, std::memory_order_relaxed);
}

inline void InterlockedAdd(uint* memory, int address, uint value)
{
  std::atomic<uint>* place = reinterpret_cast<std::atomic<uint>*>(&memory[address]);
  place->fetch_add(value, std::memory_order_relaxed);
}

inline void InterlockedAdd(float* memory, int address, float value)
{
  std::atomic<float>* place = reinterpret_cast<std::atomic<float>*>(&memory[address]);
  float current = place->load(std::memory_order_relaxed);
  float goal = current + value;
  while (!place->compare_exchange_weak(current, goal, std::memory_order_release, std::memory_order_relaxed));
}

inline int InterlockedAdd_Prev(int* memory, int address, int value)
{
  std::atomic<int>* place = reinterpret_cast<std::atomic<int>*>(&memory[address]);
  return place->fetch_add(value, std::memory_order_relaxed);
}

inline uint InterlockedAdd_Prev(uint* memory, int address, uint value)
{
  std::atomic<uint>* place = reinterpret_cast<std::atomic<uint>*>(&memory[address]);
  return place->fetch_add(value, std::memory_order_relaxed);
}

inline float InterlockedAdd_Prev(float* memory, int address, float value)
{
  std::atomic<float>* place = reinterpret_cast<std::atomic<float>*>(&memory[address]);
  float current = place->load(std::memory_order_relaxed);
  float goal = current + value;
  while (!place->compare_exchange_weak(current, goal, std::memory_order_release, std::memory_order_relaxed));
  return current;
}

inline void InterlockedAnd(int* memory, int address, int value)
{
  std::atomic<int>* place = reinterpret_cast<std::atomic<int>*>(&memory[address]);
  place->fetch_or(value, std::memory_order_relaxed);
}

inline void InterlockedAnd(uint* memory, int address, uint value)
{
  std::atomic<uint>* place = reinterpret_cast<std::atomic<uint>*>(&memory[address]);
  place->fetch_and(value, std::memory_order_relaxed);
}

inline void InterlockedOr(int* memory, int address, int value)
{
  std::atomic<int>* place = reinterpret_cast<std::atomic<int>*>(&memory[address]);
  place->fetch_or(value, std::memory_order_relaxed);
}

inline void InterlockedOr(uint* memory, int address, uint value)
{
  std::atomic<uint>* place = reinterpret_cast<std::atomic<uint>*>(&memory[address]);
  place->fetch_or(value, std::memory_order_relaxed);
}

inline void InterlockedXor(int* memory, int address, int value)
{
  std::atomic<int>* place = reinterpret_cast<std::atomic<int>*>(&memory[address]);
  place->fetch_xor(value, std::memory_order_relaxed);
}

inline void InterlockedXor(uint* memory, int address, uint value)
{
  std::atomic<uint>* place = reinterpret_cast<std::atomic<uint>*>(&memory[address]);
  place->fetch_xor(value, std::memory_order_relaxed);
}

inline void InterlockedMin(int* memory, int address, int value)
{
  std::atomic<int>* place = reinterpret_cast<std::atomic<int>*>(&memory[address]);
  int current = place->load(std::memory_order_relaxed);
  int goal = min(current, value);
  while (!place->compare_exchange_weak(current, goal, std::memory_order_release, std::memory_order_relaxed)) {
      goal = min(current, value);
  }
}

inline void InterlockedMin(float* memory, int address, float value)
{
  std::atomic<float>* place = reinterpret_cast<std::atomic<float>*>(&memory[address]);
  float current = place->load(std::memory_order_relaxed);
  float goal = min(current, value);
  while (!place->compare_exchange_weak(current, goal, std::memory_order_release, std::memory_order_relaxed)) {
      goal = min(current, value);
  }
}

inline void InterlockedMax(int* memory, int address, int value)
{
  std::atomic<int>* place = reinterpret_cast<std::atomic<int>*>(&memory[address]);
  int current = place->load(std::memory_order_relaxed);
  int goal = max(current, value);
  while (!place->compare_exchange_weak(current, goal, std::memory_order_release, std::memory_order_relaxed)) {
      goal = max(current, value);
  }
}

inline void InterlockedMax(float* memory, int address, float value)
{
  std::atomic<float>* place = reinterpret_cast<std::atomic<float>*>(&memory[address]);
  float current = place->load(std::memory_order_relaxed);
  float goal = max(current, value);
  while (!place->compare_exchange_weak(current, goal, std::memory_order_release, std::memory_order_relaxed)) {
      goal = max(current, value);
  }
}

inline uint pcg(uint v)
{
	uint state = v * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

inline float pcgf(uint v)
{
	return (float)pcg(v) / (float)0xffffffffu;
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

uint allocate(uint alloc(uint*&, uint*, uint dim), uint*& mem, std::initializer_list<uint> shape)
{
  uint* shape_arr = new uint[shape.size()];

  for (int i = 0; i < shape.size(); i++)
  {
	shape_arr[i] = shape.begin()[i];
  }

  uint off = alloc(mem, shape_arr, shape.size());

  delete[] shape_arr;

  return off;
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
	    "(uint* in, uint* out, uint* mem, uint alloc(uint*&, uint*, uint dim), "
	    "void deallocate(uint))\n"
	    "{\n";

	for (auto& i : program->kernels_) {
		Kernel* kernel = &i;
		Scope* cluster = kernel->begin_->kernel_;

		C_CodeGenerator generator;

		if (kernel->type_ == KernelType::Host) {
			generator.input_memory_index = &input_memory_index;
			generator.allocated_memories = &allocated_memories;
			generator.output_memories = &output_memories;
			generator.offset_array = false;

			generator.GenerateKernelLines(program->ir_, cluster, kernel);
			string memory_code = generator.GetFinalCode();
			
			host_code += "\n";
			host_code += AddIndent(memory_code, "  ");

			continue;
		}

		string kernel_name = "kernel_" + to_string(kernel_count);
		kernel_names.push_back(kernel_name);

		// Generate kernel
		

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

		string memory_args = "{";
		for (int d = 0; d < memory_nodes.size(); d++) {
			if (d != 0) {
				memory_args += ", ";
			}
			memory_args += memory_nodes[d]->var_name;
		}
		memory_args += "}";

		string variable_args = "{";
		for (int d = 0; d < variable_nodes.size(); d++) {
			if (d != 0) {
				variable_args += ", ";
			}
			variable_args += ReadVariable(variable_nodes[d]);
		}
		variable_args += "}";

		string shape_args = "{";
		for (int d = 0; d < i.dim; d++) {
			if (d != 0) {
				shape_args += ", ";
			}
			shape_args += "(uint)"+ReadVariable(i.shape[d]->from_->get());
		}
		shape_args += "}";

		if (i.dim == 0) //add it to host code if scalar kernel
		{
			generator.offset_name_ = "off_" + to_string(kernel_count);
			generator.variable_name_ = "var_" + to_string(kernel_count);
			generator.GenerateKernelLines(program->ir_, cluster, kernel);
			string kernel_code = generator.GetFinalCode();
			kernel->generated_code_ = kernel_code;

			host_code += "\n";
			if (memory_nodes.size() > 0) {
				host_code += "  std::vector<uint> " + generator.offset_name_ + " = " + memory_args + ";\n";
			}
			if (variable_nodes.size() > 0) {
				host_code += "  std::vector<uint> " + generator.variable_name_ + " = " + variable_args + ";\n";
			}
			host_code += "\n";
			host_code += AddIndent(kernel_code, "  ");
		}
		else
		{
			generator.GenerateKernelLines(program->ir_, cluster, kernel);
			string kernel_code = generator.GetFinalCode();
			kernel->generated_code_ = kernel_code;

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

			host_code += "\n";
			host_code += "  dispatch(" + kernel_name + ", mem, " +
			             memory_args + ", " + variable_args + ", " + shape_args + ");\n";
		}
		kernel_count++;
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
