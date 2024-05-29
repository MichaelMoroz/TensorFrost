#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../Tensor/Tensor.h"
#include "BufferManager.h"

namespace TensorFrost {

using namespace std;

extern "C" {
	struct TF_Tensor {
		Buffer* buffer;
		uint dim;
		uint* shape;
		TF_Type type;
	};

	struct DispatchInfo {
		int kernel_id;
		uint tensor_count;
		TF_Tensor* tensors;
		uint variable_count;
		uint* variables;
		uint work_group_count;
	};

	typedef TF_Tensor alloc_func(uint*, uint, TF_Type);
	typedef void dealloc_func(TF_Tensor);
	typedef uint readback_func(TF_Tensor, uint);
	typedef void writeback_func(TF_Tensor, uint, uint);
	typedef void dispatch_func(DispatchInfo);
	typedef void cpu_dispatch_func(uint* var, uint** mem, uint work_group_count);
}

using uint = unsigned int;
using main_func = void(TF_Tensor*, TF_Tensor*, alloc_func, dealloc_func, readback_func, writeback_func, dispatch_func);

int GetLinearSize(const vector<int>& shape);
vector<int> GetShape(const TF_Tensor* tensor);
int GetSize(const TF_Tensor* tensor);

class TensorMemoryManager;

class TensorMemoryManager {
public:
	BufferManager buffer_manager;

	virtual Buffer* TryGetBuffer(int size) {
		throw std::runtime_error("TryGetBuffer not implemented");
	}

	virtual void SetDataAtOffset(const TF_Tensor* buffer, int offset, const vector<uint>& data) {
		throw std::runtime_error("SetDataAtOffset not implemented");
	}

	TF_Tensor* Allocate(const vector<int>& shape, const TF_Type type = TF_Type::Float) {
		int size = GetLinearSize(shape);

		if (size == 0) {
			throw invalid_argument("Trying to allocate a tensor with size 0");
		}

		Buffer* buf = TryGetBuffer(size);
		return MakeTensor(shape, buf, type);
	}

	TF_Tensor* AllocateWithData(const vector<int>& shape, const vector<uint>& data, const TF_Type type = TF_Type::Float) {
		TF_Tensor* tensor_memory = Allocate(shape, type);
		SetDataAtOffset(tensor_memory, 0, data);
		return tensor_memory;
	}

	virtual vector<uint> Readback(const TF_Tensor* memory) = 0;
	virtual uint ReadbackValue(const TF_Tensor* memory, uint index) = 0;
	virtual void Writeback(const TF_Tensor* memory, const vector<uint>& data) = 0;
	virtual void WritebackValue(const TF_Tensor* memory, uint index, uint value) = 0;

	TF_Tensor* MakeTensor(uint* shape, uint dim, Buffer* buf, TF_Type type) {
		TF_Tensor* tensor = new TF_Tensor();
		tensor->buffer = buf;
		tensor->dim = dim;
		tensor->shape = shape;
		tensor->type = type;
		return tensor;
	}

	TF_Tensor* MakeTensor(const vector<int>& shape, Buffer* buf, TF_Type type) {
		uint* shape_arr = new uint[shape.size()];
		std::copy(shape.begin(), shape.end(), shape_arr);
		return MakeTensor(shape_arr, (int)shape.size(), buf, type);
	}

	void Free(TF_Tensor* memory) {
		buffer_manager.DeallocateBuffer(memory->buffer);
	}

	uint32_t GetAllocatedSize() const {
		return buffer_manager.GetRequiredAllocatedStorage();
	}

	uint32_t GetUnusedAllocatedSize() const {
		return buffer_manager.GetUnusedAllocatedStorage();
	}
};


extern TensorMemoryManager* global_memory_manager;

}  // namespace TensorFrost