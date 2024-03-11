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

	map<DataType, string> type_names = {
	    {DataType::None, "void"},   {DataType::Bool, "bool"},
	    {DataType::Float, "float"}, {DataType::Uint, "uint"},
	    {DataType::Int, "int"},
	};

	int* input_memory_index;

	bool offset_array = true;

	ArgumentNames GenerateArgumentNames(ArgumentMap args, map<Node*, int> variables) override {
		ArgumentNames names;
		for (auto& arg : args) {
			string name = GetNodeName(arg.second, true);
			if (variables.contains(arg.second)) {
				name = variable_name_ + "[" + to_string(variables[arg.second]) + "]";
				name =
				    "as" + type_names[arg.second->GetTensor()->type] + "(" + name + ")";
			}
			names[arg.first] = name;
		}
		return names;
	}

	Line* GenerateLine(Node* node, map<Node*, int> offsets, map<Node*, int> variables) override {
		//TODO: Create argument manager class
		ArgumentMap args = node->GetArgumentMap();
		ArgumentNames names = GenerateArgumentNames(args, variables);
		ArgumentTypes types = node->GetArgumentTypes();
		ArgumentCount arg_count = node->GetArgumentCounts();
		const Operation* op = node->op;
		string name = node->var_name;

		// get output type
		DataType output_type = node->tensor_->type;

		// lambda to get argument names
		auto ArgName = [&](ArgType type, int index = 0) {
			if (!names.contains(ArgID(type, index)))
			{
				throw std::runtime_error("Argument name not found");
			}
			return names[ArgID(type, index)];
		};

		auto HasArgument = [&](ArgType type, int index = 0) {
			return args.contains(ArgID(type, index));
		};

		auto Argument = [&](ArgType type, int index = 0) { 
			if (!args.contains(ArgID(type, index))) {
				throw std::runtime_error("Argument not found");
			}
			return args[ArgID(type, index)];
		};

		auto Type = [&](ArgType type, int index = 0) {
			return types[ArgID(type, index)];
		};


		// generate line
		string left = "";
		string expression = "";
		string right = "";
		bool needs_parenthesis = true;
		if (op->name_ == "loop") {
			left += "for (int " + name + " = " + ArgName(ArgType::Input, 0) + "; " + name +
			        " < " + ArgName(ArgType::Input, 1) + "; " + name +
			        " += " + ArgName(ArgType::Input, 2) + ")";
		}  else if (op->name_ == "if") {
			left += "if (" + ArgName(ArgType::Input, 0) + ")";
		}  else if (op->HasAllTypes(OpType::MemoryOp)) {
			string address;
			if (offset_array)
			{
				address = offset_name_ + "[" + to_string(offsets[Argument(ArgType::Memory)]) + "]";
			}
			else
			{
				address = ArgName(ArgType::Memory);
			}
			
			//if has index (not a scalar)
			if (HasArgument(ArgType::Index)) {
				address += " + " + ArgName(ArgType::Index);
			}
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
				if (Type(ArgType::Memory) != DataType::Uint) {
					expression += "asuint(";
				}
				expression += ArgName(ArgType::Input, 0);
				if (Type(ArgType::Memory) != DataType::Uint) {
					expression += ")";
				}
				right += ";";
			}
			else if (op->HasAllTypes(OpType::Scatter))
			{
				if (output_type != DataType::None) {
					left += type_names[output_type] + " " + name + " = ";
				}
				string input_type_name = type_names[Type(ArgType::Input)];
				expression += op->code_ + "((" + input_type_name + "*)mem" +
				              ", " + address + ", " + ArgName(ArgType::Input) + ")";
				right += ";";
			}
		} else if (op->name_ == "set") {
			left += ArgName(ArgType::Memory) + " = ";
			expression += ArgName(ArgType::Input);
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
				ArgMap args = node->GetArgumentMap(ArgType::Shape);
				int dims = (int)args.size();
			
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
			}
		}
		else if (op->name_ == "deallocate") {
			left = "deallocate(" + ArgName(ArgType::Memory) + ")";
			right = ";";
		} else {
			if (output_type != DataType::None) {
				left += type_names[output_type] + " " + name + " = ";
			}
			string line;

			switch (op->op_types_[0]) { //TODO: properly support multiple op types
				case OpType::Operator:
					line += ArgName(ArgType::Input, 0) + " " + op->code_ + " " +
					        ArgName(ArgType::Input, 1);
					break;
				case OpType::UnaryOperator:
					line += op->code_ + ArgName(ArgType::Input, 0);
					break;
				case OpType::Function:
					line += op->code_ + "(";
					for (int i = 0; i < arg_count[ArgType::Input]; i++) {
						if (i != 0) {
							line += ", ";
						}
						line += ArgName(ArgType::Input, i);
					}
					line += ")";
					needs_parenthesis = false;
					break;
				case OpType::Keyword:
					line += op->code_;
					break;
				case OpType::DimensionIndex:
					line += op->code_ + to_string(node->GetTensor()->data[0]);
					needs_parenthesis = false;
					break;
				case OpType::TypeCast:
					line += "(" + op->code_ + ")" + ArgName(ArgType::Input, 0);
					break;
				case OpType::TypeReinterpret:
					line += "*(" + op->code_ + "*)&" + ArgName(ArgType::Input, 0);
					break;
				case OpType::Constant:
					line += node->GetTensor()->GetConstantString();
					needs_parenthesis = false;
					break;
				case OpType::TernaryOperator:
					line += ArgName(ArgType::Input, 0) + " ? " + ArgName(ArgType::Input, 1) +
					        " : " + ArgName(ArgType::Input, 2);
					break;
				default:
					line += "";
					break;
			}
			expression += line;
			right += ";";
		}

		return new Line(left, expression, right, name, needs_parenthesis, op->cost_);
	}
};

string GenerateC(Program* program) {
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

inline uint asuint(int x)
{
  return *(uint*)&x;
}

inline uint asuint(uint x)
{
  return *(uint*)&x;
}

inline int asint(uint x)
{
  return *(int*)&x;
}

inline int clamp(int x, int a, int b)
{
  return min(max(x, a), b);
}

inline float clamp(float x, float a, float b)
{
  return min(max(x, a), b);
}

inline float lerp(float a, float b, float t)
{
  return a + (b - a) * t;
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
  while (!place->compare_exchange_weak(current, goal, std::memory_order_release, std::memory_order_relaxed)) {
      goal = current + value;
  }
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
  while (!place->compare_exchange_weak(current, goal, std::memory_order_release, std::memory_order_relaxed)) {
      goal = current + value;
  }
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
	int input_memory_index = 0;

	// Generate code for each compute kernel
	int kernel_count = 0;
	map<Node*, string> dispatch_code;

	for (auto& i : program->kernels_) {
		Kernel* kernel = &i;
		Node* kernel_node = kernel->kernel_node_;

		string kernel_name = "kernel_" + to_string(kernel_count++);

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
			variable_args += "asuint(" + ReadVariable(variable_nodes[d]) + ")";
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

		C_CodeGenerator generator;
		generator.GenerateKernelLines(program->ir_, kernel_node, kernel);
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
			"void " +
			kernel_name +
			"(uint* var, uint* off, uint* mem, uint* shape)\n"
			"{\n"
			"  #pragma omp parallel for shared(mem) \n" +
			loop + AddIndent(kernel_code, "    ") +
			"  }\n"
			"}\n";

		dispatch_code[kernel_node] = "dispatch(" + kernel_name + ", mem, " + memory_args + ", " + variable_args + ", " + shape_args + ")";
	}

	C_CodeGenerator generator;
	generator.custom_generated_code_ = dispatch_code;
	generator.input_memory_index = &input_memory_index;
	generator.offset_array = false;
	generator.GenerateKernelLines(program->ir_, program->ir_->root, &program->kernels_[0]);

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

	host_code += AddIndent(generator.GetFinalCode(), "  ");

	//set output memories and deallocate
	for (auto& memory : program->ir_->output_memory_map) {
		int output_memory_index = memory.first;
		string mem_name = memory.second->var_name;
		host_code += "  out[" + to_string(output_memory_index) + "] = " + mem_name + ";\n";
	}

	all_kernels += host_code + "}\n";

	program->generated_code_ = all_kernels;
	return all_kernels;
}
}  // namespace TensorFrost
