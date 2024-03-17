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
		Node* kernel_node = kernel->node_;

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

		CodeGenerator generator;
		generator.GenerateKernelCode(kernel);
		string kernel_code = generator.AssembleString();
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


		final_source +=
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

	CodeGenerator generator;
	generator.custom_generated_code_ = dispatch_code;
	generator.input_memory_index = &input_memory_index;
	generator.offset_array = false;
	generator.GenerateCode(program->ir_->root);

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

	host_code += AddIndent(generator.AssembleString(), "  ");

	//set output memories and deallocate
	for (auto& memory : program->ir_->output_memory_map) {
		int output_memory_index = memory.first;
		string mem_name = memory.second->var_name;
		host_code += "  out[" + to_string(output_memory_index) + "] = " + mem_name + ";\n";
	}

	final_source += host_code + "}\n";

	return final_source;
}
}  // namespace TensorFrost
