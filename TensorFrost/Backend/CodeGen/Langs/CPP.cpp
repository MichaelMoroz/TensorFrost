#pragma once

#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;

string GenerateCPP(Program* program) {
	string final_source = R"(
#include <cmath>
#include <omp.h>
#include <initializer_list>
#include <functional>
#include <vector>
#include <atomic>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

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

extern "C" {
	enum DataType {
		Float,
		Uint,
		Int,
		Bool,
		None,
	};

	struct TensorProp {
		uint offset;
		uint dim;
		uint* shape;
		DataType type;
	};

	typedef TensorProp alloc_func(uint*&, uint*, uint, DataType);
	typedef void dealloc_func(TensorProp);
	typedef void kernel_func(uint*, uint*, uint*, uint*);
}

std::unordered_map<DataType, std::string> DataTypeNames = {
    {DataType::Float, "Float"}, {DataType::Uint, "Uint"},
    {DataType::Int, "Int"},     {DataType::Bool, "Bool"},
    {DataType::None, "None"},
};

uint* mem;
alloc_func* alloc;
dealloc_func* dealloc;

void dispatch(kernel_func* kernel, std::initializer_list<TensorProp> tensors, std::initializer_list<uint> var, std::initializer_list<uint> shape)
{
  uint* off_arr = new uint[tensors.size()];
  uint* var_arr = new uint[var.size()];
  uint* shape_arr = new uint[shape.size()];

  for (int i = 0; i < tensors.size(); i++)
  {
    off_arr[i] = tensors.begin()[i].offset;
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

TensorProp allocate(std::initializer_list<uint> shape, DataType type)
{
  uint* shape_arr = new uint[shape.size()];

  for (int i = 0; i < shape.size(); i++)
  {
	shape_arr[i] = shape.begin()[i];
  }

  TensorProp tensor = alloc(mem, shape_arr, shape.size(), type);

  delete[] shape_arr;

  return tensor;
}

void deallocate(TensorProp tensor)
{
  dealloc(tensor);
}

TensorProp check_tensor(TensorProp tensor, std::string name, std::initializer_list<uint> shape, DataType type)
{
	if (tensor.type != type)
	{
		throw std::runtime_error("Invalid type for " + name + ". Expected " + DataTypeNames[type] + ", got " + DataTypeNames[tensor.type]);
	}

	if (tensor.dim != shape.size())
	{
		throw std::runtime_error("Invalid number of dimensions for " + name + ". Expected " + std::to_string(shape.size()) + ", got " + std::to_string(tensor.dim));
	}

	uint* shape_arr = tensor.shape;
	for (int i = 0; i < tensor.dim; i++)
	{
		int shape_val = shape.begin()[i];
		if (shape_arr[i] != shape_val || shape_val < 1)
		{
			throw std::runtime_error("Invalid shape for dimension " + std::to_string(i) + " in " + name + ". Expected " + std::to_string(shape_val) + ", got " + std::to_string(shape_arr[i]));
		}
	}

	return tensor;
}

)";
	
	GenerateNodeNames(*program->ir_);
	int input_count = program->ir_->memory_inputs.size();
	int output_count = program->ir_->output_memory_map.size();

	// Generate code for each compute kernel
	int kernel_count = 0;
	map<Node*, string> dispatch_code;

	for (auto& i : program->kernels_) {
		Kernel* kernel = &i;

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

		final_source += GenerateCPPKernel(program, kernel, kernel_name);

		dispatch_code[kernel->root] = "dispatch(" + kernel_name + ", " + memory_args + ", " + variable_args + ", " + shape_args + ")";
	}

	CodeGenerator generator;
	generator.custom_generated_code_ = dispatch_code;
	generator.offset_array = false;
	generator.GenerateCode(program->ir_->root);


	string main_code = "\nstd::tuple<";
	for (int i = 0; i < output_count; i++) {
		main_code += "TensorProp";
		if (i != output_count - 1) {
			main_code += ", ";
		}
	}
	main_code += "> " + program->program_name + "(";

	for (int i = 0; i < input_count; i++) {
		main_code += "TensorProp in" + to_string(i);
		if (i != input_count - 1) {
			main_code += ", ";
		}
	}
	main_code += ")\n{\n";

	main_code += AddIndent(generator.AssembleString(), "  ");

	main_code += "  return {";

	for (int i = 0; i < output_count; i++) {
		Node* output_node = program->ir_->output_memory_map[i];
		main_code += output_node->var_name;
		if (i != output_count - 1) {
			main_code += ", ";
		}
	}
	main_code += "};\n}\n";

	final_source += main_code;

	string host_code =
	    "\n"
	    "extern \"C\" "
#ifdef _WIN32
	    "__declspec(dllexport)"
#endif
	    " void "
	    "main"
	    "(TensorProp* in, TensorProp* out, uint* mem_address, alloc_func "
	    "allocation, "
	    "dealloc_func deallocation)\n"
	    "{\n"
	    "  mem = mem_address;\n"
	    "  alloc = allocation;\n"
	    "  dealloc = deallocation;\n"
		"  auto outputs = " + program->program_name + "(";

	for (int i = 0; i < input_count; i++) {
		host_code += "in[" + to_string(i) + "]";
		if (i != input_count - 1) {
			host_code += ", ";
		}
	}
	host_code += ");\n";

	for (int i = 0; i < output_count; i++) {
		host_code += "  out[" + to_string(i) + "] = std::get<" + to_string(i) + ">(outputs);\n";
	}

	host_code += "}\n";

	final_source += host_code;

	return final_source;
}

string GenerateCPPKernel(Program* program, const Kernel* kernel, const string& kernel_name) {
	CodeGenerator generator;
	generator.GenerateKernelCode(kernel);
	string kernel_code = generator.AssembleString();

	string loop = "";
	const int block_size = 4;  // TODO chose automatically
	switch (kernel->indexing_mode_) {
		case KernelIndexingMode::Linear:
			loop = "  for (int thread_id = 0; thread_id < shape[0]; thread_id++)\n";
			loop += "  {\n";
			break;
		case KernelIndexingMode::MultiDimensional:
			for (int d = 0; d < kernel->dim; d++) {
				loop += "  for (int dim" + to_string(d) + " = 0; dim" + to_string(d) +
				        " < shape[" + to_string(d) + "]; dim" + to_string(d) + "++)\n";
			}
			loop += "  {\n";
			break;
		case KernelIndexingMode::MultiDimensionalBlocks:
			for (int d = 0; d < kernel->dim; d++) {
				loop += "  for (int wg" + to_string(d) + " = 0; wg" + to_string(d) +
				        " < shape[" + to_string(d) + "]; wg" + to_string(d) +
				        "+= " + to_string(block_size) + ")\n";
			}
			for (int d = 0; d < kernel->dim; d++) {
				loop += "  for (int dim" + to_string(d) + " = wg" + to_string(d) +
				        "; dim" + to_string(d) + " < min(wg" + to_string(d) + "+" +
				        to_string(block_size) + ", shape[" + to_string(d) + "]); dim" +
				        to_string(d) + "++)\n";
			}
			loop += "  {\n";
			break;
		default:
			throw std::runtime_error("Invalid indexing mode");
			break;
	}

	string kernel_source =
	    "\n"
	    "void " +
	    kernel_name +
	    "(uint* var, uint* off, uint* mem, uint* shape)\n"
	    "{\n"
	    "  #pragma omp parallel for shared(mem) \n" +
	    loop + AddIndent(kernel_code, "    ") +
	    "  }\n"
	    "}\n";

	return kernel_source;
}

}  // namespace TensorFrost
