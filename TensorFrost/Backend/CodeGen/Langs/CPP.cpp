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

typedef uint32_t uint;

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

inline uint asuint(bool x)
{
	return *(uint*)&x;
}

inline bool asbool(uint x)
{
	return *(bool*)&x;
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
	enum class TFType {
		Float,
		Uint,
		Int,
		Bool,
		None,
	};

	struct TFBuffer {
		size_t size = 0;
		size_t used_size = 0;
		size_t time_since_used = 0;
		bool up_to_date = false;
		bool read_only = false;
		const char* name = nullptr;
		//add type descriptor (for special kinds of buffers)
	};

	struct TFTensor {
		TFBuffer* buffer;
		TFType type;
		size_t dim;
		const size_t* shape;
	};

	struct TFTensorList {
		size_t count;
		const TFTensor* tensors;
	};

	struct TFDispatchInfo {
		size_t kernel_id;
		size_t read_write_count;
		const TFTensor* read_write_tensors;
		size_t read_only_count;
		const TFTensor* read_only_tensors;
		size_t variable_count;
		const uint32_t* variables;
		size_t work_group_count;
	};

	typedef TFTensor alloc_func(const char*, const size_t*, size_t, TFType, void*);
	typedef void dealloc_func(TFTensor, void*);
	typedef uint readback_func(TFTensor, size_t, void*);
	typedef void writeback_func(TFTensor, size_t, uint32_t, void*);
	typedef void dispatch_func(TFDispatchInfo, void*);

	struct TFRuntime {
		alloc_func* alloc;
		dealloc_func* dealloc;
		readback_func* readback;
		writeback_func* writeback;
		dispatch_func* dispatch;
		void* custom_data;
	};

}

class TFContext
{
public:
	TFRuntime runtime;

	TFContext(TFRuntime runtime);
	size_t compute_size(const size_t* shape, size_t dim);
	TFTensor allocate(std::string name, std::initializer_list<size_t> shape, TFType type);
	void deallocate(TFTensor tensor);
	void check_tensor(TFTensor tensor, std::string name, std::initializer_list<size_t> target_shape, TFType target_type);
	TFTensor reshape(TFTensor tensor, std::string name, std::initializer_list<size_t> shape, TFType type);
	TFTensor assert(TFTensor tensor, std::string name, std::initializer_list<size_t> target_shape, TFType target_type);
	uint32_t read(TFTensor tensor, size_t index);
	void write(TFTensor tensor, size_t index, uint32_t value);
	void dispatch(size_t kernel_id, std::initializer_list<TFTensor> read_write, std::initializer_list<TFTensor> read_only, std::initializer_list<uint32_t> var, std::initializer_list<size_t> shape, std::initializer_list<size_t> group);
};
)";

	return header;
}

string GetCPPImplementation() {
	string implementation = R"(

std::unordered_map<TFType, std::string> TFTypeNames = {
    {TFType::Float, "Float"}, {TFType::Uint, "Uint"},
    {TFType::Int, "Int"},     {TFType::Bool, "Bool"},
    {TFType::None, "None"},
};

TFContext::TFContext(TFRuntime runtime) : runtime(runtime) {}

size_t TFContext::compute_size(const size_t* shape, size_t dim) {
	size_t size = 1;
	for (size_t i = 0; i < dim; i++) {
	  size *= shape[i];
	}
	return size;
}

TFTensor TFContext::allocate(std::string name, std::initializer_list<size_t> shape, TFType type)
{
	const size_t* shape_arr = shape.begin();
	size_t dim = shape.size();
	size_t size = compute_size(shape_arr, dim);

	for (size_t i = 0; i < dim; i++) {
		if(shape_arr[i] < 1) {
			throw std::runtime_error("Invalid shape on dimension " + std::to_string(i) + " for " + name + ". Expected positive integer, got " + std::to_string(shape_arr[i]));
		}
	}

	return runtime.alloc(name.c_str(), shape_arr, dim, type, runtime.custom_data);
}

void TFContext::deallocate(TFTensor tensor)
{
	runtime.dealloc(tensor, runtime.custom_data);
}

void TFContext::check_tensor(TFTensor tensor, std::string name, std::initializer_list<size_t> target_shape, TFType target_type)
{
	const size_t* shape_arr = tensor.shape;
	const size_t* target_shape_arr = target_shape.begin();
	size_t target_dim = target_shape.size();

	if (tensor.type != target_type) {
		throw std::runtime_error("Invalid type for " + name + ". Expected " + TFTypeNames[target_type] + ", got " + TFTypeNames[tensor.type]);
	}

	if (tensor.dim != target_dim) {
		throw std::runtime_error("Invalid number of dimensions for " + name + ". Expected " + std::to_string(target_dim) + ", got " + std::to_string(tensor.dim));
	}

	for (size_t i = 0; i < tensor.dim; i++) {
		if (shape_arr[i] != target_shape_arr[i] || target_shape_arr[i] < 1) {
			throw std::runtime_error("Invalid shape for dimension " + std::to_string(i) + " in " + name + ". Expected " + std::to_string(target_shape_arr[i]) + ", got " + std::to_string(shape_arr[i]));
		}
	}
}

TFTensor TFContext::reshape(TFTensor tensor, std::string name, std::initializer_list<size_t> shape, TFType type)
{
	size_t* new_shape = new size_t[shape.size()];
	std::copy(shape.begin(), shape.end(), new_shape);
	TFTensor new_tensor = {tensor.buffer, type, shape.size(), new_shape};

	size_t old_size = compute_size(tensor.shape, tensor.dim);
	size_t new_size = compute_size(new_tensor.shape, new_tensor.dim);

	if(old_size != new_size) {
		throw std::runtime_error("Cannot reshape " + name + ", expected " + std::to_string(new_size) + " elements, while input has " + std::to_string(old_size));
	}

	return new_tensor;
}

TFTensor TFContext::assert(TFTensor tensor, std::string name, std::initializer_list<size_t> target_shape, TFType target_type)
{
	check_tensor(tensor, name, target_shape, target_type);
	return tensor;
}

uint TFContext::read(TFTensor tensor, size_t index)
{
	return runtime.readback(tensor, index, runtime.custom_data);
}

void TFContext::write(TFTensor tensor, size_t index, uint32_t value)
{
	runtime.writeback(tensor, index, value, runtime.custom_data);
}

void TFContext::dispatch(size_t kernel_id, std::initializer_list<TFTensor> read_write, std::initializer_list<TFTensor> read_only, std::initializer_list<uint32_t> var, std::initializer_list<size_t> shape, std::initializer_list<size_t> group)
{
	//currently only supports read_write tensors
	std::vector<TFTensor> all_tensors;
	all_tensors.insert(all_tensors.end(), read_write.begin(), read_write.end());
	all_tensors.insert(all_tensors.end(), read_only.begin(), read_only.end());
	TFDispatchInfo info = {kernel_id, all_tensors.size(), all_tensors.data(), 0, nullptr, (uint)var.size(), var.begin(), 0};

	const TFTensor* read_write_tensors = read_write.begin();
	for (size_t i = 0; i < read_write.size(); i++) {
		read_write_tensors[i].buffer->up_to_date = false;
	}

	const size_t* shape_arr = shape.begin();
	const size_t* group_arr = group.begin();
	size_t dispatch_dim = shape.size();
	size_t group_dim = group.size();

	size_t work_group_count = 1;
	for (size_t i = 0; i < dispatch_dim - group_dim; i++) {
		work_group_count *= shape_arr[i];
	}

	//only the last dimensions are divided by the group size
	for (size_t i = 0; i < group_dim; i++) {
		size_t dim = shape_arr[dispatch_dim - group_dim + i];
		work_group_count *= (dim + group_arr[i] - 1) / group_arr[i];
	}

	info.work_group_count = work_group_count;

	runtime.dispatch(info, runtime.custom_data);
}
)";
	return implementation;
}

void GenerateCode(Program* program) {
	string final_source = GetCPPHeader();
	final_source += GetCPPImplementation();

	GenerateNodeNames(*program->ir_);
	int input_count = (int)program->ir_->input_memory_map.size();
	int output_count = (int)program->ir_->output_memory_map.size();

	// Generate code for each compute kernel
	map<Node*, string> dispatch_code;

	for (auto& kernel : program->kernels_) {
		global_kernel_manager->AddKernelID(program, &kernel);
		kernel.kernel_name_ = "kernel_" + to_string(kernel.kernel_id_);

		// Generate kernel
		vector<Node*> read_write_nodes;
		read_write_nodes.resize(kernel.read_write_memory.size());
		for (auto& read_write : kernel.read_write_memory) {
			read_write_nodes[read_write.second] = read_write.first;
		}
		vector<Node*> read_only_nodes;
		read_only_nodes.resize(kernel.read_only_memory.size());
		for (auto& read_only : kernel.read_only_memory) {
			read_only_nodes[read_only.second] = read_only.first;
		}

		vector<Node*> variable_nodes;
		variable_nodes.resize(kernel.variables.size());
		for (auto& variable : kernel.variables) {
			variable_nodes[variable.second] = variable.first;
		}

		string read_write_args = "{";
		for (int d = 0; d < read_write_nodes.size(); d++) {
			if (d != 0) {
				read_write_args += ", ";
			}
			read_write_args += read_write_nodes[d]->var_name;
		}
		read_write_args += "}";
		string read_only_args = "{";
		for (int d = 0; d < read_only_nodes.size(); d++) {
			if (d != 0) {
				read_only_args += ", ";
			}
			read_only_args += read_only_nodes[d]->var_name;
		}
		read_only_args += "}";

		string variable_args = "{";
		for (int d = 0; d < variable_nodes.size(); d++) {
			if (d != 0) {
				variable_args += ", ";
			}
			variable_args += "asuint(" + ReadVariable(variable_nodes[d]) + ")";
		}
		variable_args += "}";

		string shape_args = "{";
		for (int d = 0; d < kernel.shape.size(); d++) {
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
			final_source += kernel.full_generated_code_;
		}

		dispatch_code[kernel.root] = "tf.dispatch(" + to_string(kernel.kernel_id_) + ", " + read_write_args + ",  " + read_only_args + ", " + variable_args + ", " + shape_args + ", " + group_args + ")";
	}

	GenerateMain(program, dispatch_code, input_count, output_count);

	final_source += program->main_function_;

	string host_code =
	    "\n"
	    "extern \"C\" "
#ifdef _WIN32
	    "__declspec(dllexport) "
#endif
	    "int "
	    "main"
	    "(TFTensor* in, TFTensor* out, TFRuntime runtime)\n"
	    "{\n"
		"  auto outputs = " + program->program_name + "(TFContext(runtime)";

	if (input_count > 0) {
		host_code += ", ";
	}

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
	CodeGenerator generator = CodeGenerator(program->ir_);
	generator.custom_generated_code_ = dispatch_code;
	generator.GenerateCode(program->ir_->root);

	string main_code = "\nstd::tuple<";
	for (int i = 0; i < output_count; i++) {
		main_code += "TFTensor";
		if (i != output_count - 1) {
			main_code += ", ";
		}
	}
	main_code += "> " + program->program_name + "(TFContext tf";
	if (input_count > 0) {
		main_code += ", ";
	}
	for (int i = 0; i < input_count; i++) {
		Node* input_node = program->ir_->input_memory_map[i];
		main_code += "TFTensor " + input_node->var_name;
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
	CodeGenerator generator = CodeGenerator(program->ir_);
	generator.GenerateKernelCode(kernel);
	string kernel_code = generator.AssembleString();

	string loop = "";
	loop += GetBufferDeclarations(kernel, [](const string& name, const string& type_name, size_t binding) {
		return "  uint* " + name + "_mem = mem[" + to_string(binding) + "];\n";
	});

	kernel->var_names = vector<string>(kernel->variables.size());
	kernel->var_types = vector<string>(kernel->variables.size());
	for (auto var : kernel->variables) {
		kernel->var_names[var.second] = var.first->var_name;
		kernel->var_types[var.second] = type_names[var.first->type];
	}
	for (int i = 0; i < kernel->var_names.size(); i++) {
		loop += "  " + kernel->var_types[i] + " var_" + kernel->var_names[i] + " = as" + kernel->var_types[i] + "(var[" + to_string(i) + "]);\n";
	}

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

	kernel->full_generated_code_ = kernel_source;
	kernel->generated_header_ = "";
	kernel->generated_bindings_ = "";
	kernel->generated_main_ = kernel_source;
}

}  // namespace TensorFrost