#include "Backend/CodeGen/Generators.h"

namespace TensorFrost {
using namespace std;



string GetCPPHeader() {
	string header = R"(
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

inline float smoothstep(float a, float b, float t)
{
  t = clamp((t - a) / (b - a), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

inline float sign(float x)
{
  return x < 0.0f ? -1.0f : 1.0f;
}

inline int sign(int x)
{
  return x < 0 ? -1 : 1;
}

inline uint reversebits(uint x)
{
  x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
  x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
  x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
  x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
  return ((x >> 16) | (x << 16));
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
	struct Buffer {
		int size = 0;
    };

	enum DataType {
		Float,
		Uint,
		Int,
		Bool,
		None,
	};

	struct TensorProp {
		Buffer* buffer;
		uint dim;
		uint* shape;
		DataType type;
	};

	struct DispatchInfo {
		int kernel_id;
		uint tensor_count;
		TensorProp* tensors;
		uint variable_count;
		uint* variables;
		uint work_group_count;
	};

	typedef TensorProp alloc_func(uint*, uint, DataType);
	typedef void dealloc_func(TensorProp);
	typedef uint readback_func(TensorProp, uint);
	typedef void writeback_func(TensorProp, uint, uint);
	typedef void dispatch_func(DispatchInfo);
	typedef void cpu_dispatch_func(uint* var, uint** mem, uint work_group_count);
}

std::unordered_map<DataType, std::string> DataTypeNames = {
    {DataType::Float, "Float"}, {DataType::Uint, "Uint"},
    {DataType::Int, "Int"},     {DataType::Bool, "Bool"},
    {DataType::None, "None"},
};

alloc_func* alloc;
dealloc_func* dealloc;
readback_func* readback;
writeback_func* writeback;
dispatch_func* dispatch_ref;

TensorProp allocate(std::string name, std::initializer_list<uint> shape, DataType type)
{
  uint* shape_arr = new uint[shape.size()];
  uint size = 1;

  for (int i = 0; i < shape.size(); i++)
  {
	int shape_val = shape.begin()[i];
	if(shape_val < 1)
	{
		throw std::runtime_error("Invalid shape on dimension " + std::to_string(i) + " for " + name + ". Expected positive integer, got " + std::to_string(shape_val));
	}
    shape_arr[i] = shape_val;
	size *= shape_arr[i];
  }

  TensorProp tensor = alloc(shape_arr, shape.size(), type);

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

TensorProp reshape(TensorProp tensor, std::string name, std::initializer_list<uint> shape, DataType type)
{
  TensorProp new_tensor = TensorProp();
  new_tensor.buffer = tensor.buffer;
  new_tensor.dim = shape.size();
  new_tensor.shape = new uint[shape.size()];
  new_tensor.type = type;

  int old_size = 1;
  for (int i = 0; i < tensor.dim; i++)
  {
	old_size *= tensor.shape[i];
  }
  int new_size = 1;
  for (int i = 0; i < shape.size(); i++)
  {
	new_tensor.shape[i] = shape.begin()[i];
	new_size *= new_tensor.shape[i];
  }
  if(old_size != new_size)
  {
	throw std::runtime_error("Cannot reshape " + name + ", expected " + std::to_string(new_size) + " elements, while input has " + std::to_string(old_size));
  }

  return new_tensor;
}

uint ReadFromMemory(TensorProp tensor, uint index)
{
  return readback(tensor, index);
}

void WriteToMemory(TensorProp tensor, uint index, uint value)
{
  writeback(tensor, index, value);
}

void dispatch(int kernel_id, std::initializer_list<TensorProp> tensors, std::initializer_list<uint> var, std::initializer_list<uint> shape, std::initializer_list<int> group)
{
  DispatchInfo info;
  info.kernel_id = kernel_id;
  info.tensor_count = tensors.size();
  info.tensors = new TensorProp[tensors.size()];
  info.variable_count = var.size();
  info.variables = new uint[var.size()];

  int dispatch_dim = shape.size();
  int group_dim = group.size();

  int work_group_count = 1;
  for (int i = 0; i < dispatch_dim - group_dim; i++) {
  	int dim = shape.begin()[i];
  	work_group_count *= dim;
  }
  //only the last dimensions are divided by the group size
  for (int i = 0; i < group_dim; i++) {
  	int dim = shape.begin()[dispatch_dim - group_dim + i];
  	dim = (dim + group.begin()[i] - 1) / group.begin()[i];
  	work_group_count *= dim;
  }

  info.work_group_count = work_group_count;

  for (int i = 0; i < tensors.size(); i++)
  {
  	info.tensors[i] = tensors.begin()[i];
  }

  for (int i = 0; i < var.size(); i++)
  {
  	info.variables[i] = var.begin()[i];
  }

  dispatch_ref(info);

  delete[] info.tensors;
  delete[] info.variables;
} 

)";

	return header;
}


void GenerateCode(Program* program) {
	string final_source = GetCPPHeader();

	GenerateNodeNames(*program->ir_);
	int input_count = (int)program->ir_->memory_inputs.size();
	int output_count = (int)program->ir_->output_memory_map.size();

	// Generate code for each compute kernel
	map<Node*, string> dispatch_code;

	for (auto& kernel : program->kernels_) {
		global_kernel_manager->AddKernelID(program, &kernel);
		kernel.kernel_name_ = "kernel_" + to_string(kernel.kernel_id_);

		// Generate kernel
		vector<Node*> memory_nodes;
		memory_nodes.resize(kernel.memory.size());
		for (auto& memory : kernel.memory) {
			memory_nodes[memory.second] = memory.first;
		}

		vector<Node*> variable_nodes;
		variable_nodes.resize(kernel.variables.size());
		for (auto& variable : kernel.variables) {
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
		for (int d = 0; d < kernel.dim; d++) {
			if (d != 0) {
				shape_args += ", ";
			}
			shape_args += "(uint)" + ReadVariable(kernel.shape[ArgID(ArgType::Shape,d)]);
		}
		shape_args += "}";

		string group_args = "{";
		for (int d = 0; d < kernel.root->group_size.size(); d++) {
			if (d != 0) {
				group_args += ", ";
			}
			group_args += to_string(kernel.root->group_size[d]);
		}
		group_args += "}";

		GenerateKernel(program, &kernel);

		if (current_backend == BackendType::CPU) {
			final_source += kernel.generated_code_;
		}

		dispatch_code[kernel.root] = "dispatch(" + to_string(kernel.kernel_id_) + ", " + memory_args + ", " + variable_args + ", " + shape_args + ", " + group_args + ")";
	}

	GenerateMain(program, dispatch_code, input_count, output_count);

	final_source += program->main_function_;

	string host_code =
	    "\n"
	    "extern \"C\" "
#ifdef _WIN32
	    "__declspec(dllexport)"
#endif
	    "int "
	    "main"
	    "(TensorProp* in, TensorProp* out, alloc_func alloc_, dealloc_func dealloc_, readback_func readback_, writeback_func writeback_, dispatch_func dispatch_)\n"
	    "{\n"
	    "  alloc = alloc_;\n"
	    "  dealloc = dealloc_;\n"
		"  readback = readback_; \n"
		"  writeback = writeback_; \n"
		"  dispatch_ref = dispatch_; \n"
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

	host_code += "  return 0;\n}\n";

	final_source += host_code;

	program->generated_code_ = final_source;
}
void GenerateMain(Program* program, map<Node*, string>& dispatch_code,
                  int input_count, int output_count) {
	CodeGenerator generator;
	generator.custom_generated_code_ = dispatch_code;
	generator.is_kernel = false;
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

	program->main_function_ = main_code;
}

void GenerateCPPKernel(Program* program, Kernel* kernel) {
	CodeGenerator generator;
	generator.GenerateKernelCode(kernel);
	string kernel_code = generator.AssembleString();

	string loop = "";
	loop += GetBufferDeclarations(kernel, [](const string& name, const string& type_name, int binding) {
		return "  uint* " + name + "_mem = mem[" + to_string(binding) + "];\n";
	});

	const int block_size = 4;
	loop += "  #pragma omp parallel for\n";
	loop += "  for (int block_id = 0; block_id < work_group_count; block_id++)\n";
	loop += "  {\n";
	for (int d = 0; d < kernel->root->group_size.size(); d++) {
		int dim = (int)kernel->root->group_size.size() - d - 1;
		loop += "    for (int block_thread_id" + to_string(dim) +
		        " = 0; block_thread_id" + to_string(dim) + " < " +
		        to_string(kernel->root->group_size[d]) + "; block_thread_id" +
		        to_string(dim) + "++)\n";
	}
	loop += "    {\n";
	string loop_end = "    }\n";
	loop_end += "  }\n";
	loop += AddIndent(kernel_code, "      ");
	loop += loop_end;

	string kernel_source =
	    "\n"
	    "extern \"C\" "
		#ifdef _WIN32
		"__declspec(dllexport) "
		#endif
	    "void " +
	    kernel->kernel_name_ +
	    "(uint* var, uint** mem, uint work_group_count)\n"
	    "{\n" + loop +
	    "}\n";

	kernel->generated_code_ = kernel_source;
}

}  // namespace TensorFrost