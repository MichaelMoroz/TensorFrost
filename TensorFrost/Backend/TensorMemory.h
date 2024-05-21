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

using uint = unsigned int;
using main_func = void(TensorProp*, TensorProp*, alloc_func, dealloc_func, readback_func, writeback_func, dispatch_func);

int GetLinearSize(const vector<int>& shape);
vector<int> GetShape(const TensorProp* tensor);
int GetSize(const TensorProp* tensor);

class TensorMemoryManager;

class TensorMemoryManager {
public:
	BufferManager buffer_manager;

	virtual Buffer* TryGetBuffer(int size) {
		throw std::runtime_error("TryGetBuffer not implemented");
	}

	virtual void SetDataAtOffset(const TensorProp* buffer, int offset, const vector<uint>& data) {
		throw std::runtime_error("SetDataAtOffset not implemented");
	}

	TensorProp* Allocate(const vector<int>& shape, const DataType type = DataType::Float) {
		int size = GetLinearSize(shape);

		if (size == 0) {
			throw invalid_argument("Trying to allocate a tensor with size 0");
		}

		Buffer* buf = TryGetBuffer(size);
		return MakeTensor(shape, buf, type);
	}

	TensorProp* AllocateWithData(const vector<int>& shape, const vector<uint>& data, const DataType type = DataType::Float) {
		TensorProp* tensor_memory = Allocate(shape, type);
		SetDataAtOffset(tensor_memory, 0, data);
		return tensor_memory;
	}

	virtual vector<uint> Readback(const TensorProp* memory) = 0;
	virtual uint ReadbackValue(const TensorProp* memory, uint index) = 0;
	virtual void Writeback(const TensorProp* memory, const vector<uint>& data) = 0;
	virtual void WritebackValue(const TensorProp* memory, uint index, uint value) = 0;

	TensorProp* MakeTensor(uint* shape, uint dim, Buffer* buf, DataType type) {
		TensorProp* tensor = new TensorProp();
		tensor->buffer = buf;
		tensor->dim = dim;
		tensor->shape = shape;
		tensor->type = type;
		return tensor;
	}

	TensorProp* MakeTensor(const vector<int>& shape, Buffer* buf, DataType type) {
		uint* shape_arr = new uint[shape.size()];
		std::copy(shape.begin(), shape.end(), shape_arr);
		return MakeTensor(shape_arr, (int)shape.size(), buf, type);
	}

	void Free(TensorProp* memory) {
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