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
	struct TFTensor {
		TFBuffer* buffer;
		uint dim;
		const uint* shape;
		TFType type;
	};

	struct TFDispatchInfo {
		size_t kernel_id;
		uint tensor_count;
		const TFTensor* tensors;
		uint variable_count;
		const uint* variables;
		uint work_group_count;
	};

	typedef TFTensor alloc_func(const uint*, uint, TFType, void*);
	typedef void dealloc_func(TFTensor, void*);
	typedef uint readback_func(TFTensor, uint, void*);
	typedef void writeback_func(TFTensor, uint, uint, void*);
	typedef void dispatch_func(TFDispatchInfo, void*);
	typedef void cpu_dispatch_func(const uint* var, uint** mem, uint work_group_count);

	struct TFRuntime {
		alloc_func* alloc;
		dealloc_func* dealloc;
		readback_func* readback;
		writeback_func* writeback;
		dispatch_func* dispatch;
		void* custom_data;
	};
}

using uint = unsigned int;
using main_func = void(TFTensor*, TFTensor*, TFRuntime);

int GetLinearSize(const vector<int>& shape);
vector<int> GetShape(const TFTensor* tensor);
int GetSize(const TFTensor* tensor);

class TensorMemoryManager;

class TensorMemoryManager {
public:
	BufferManager buffer_manager;

	virtual TFBuffer* TryGetBuffer(int size) {
		throw std::runtime_error("TryGetBuffer not implemented");
	}

	virtual void SetDataAtOffset(const TFTensor* buffer, int offset, const vector<uint>& data) {
		throw std::runtime_error("SetDataAtOffset not implemented");
	}

	TFTensor* Allocate(const vector<int>& shape, const TFType type = TFType::Float) {
		int size = GetLinearSize(shape);

		if (size == 0) {
			throw invalid_argument("Trying to allocate a tensor with size 0");
		}

		TFBuffer* buf = TryGetBuffer(size);
		return MakeTensor(shape, buf, type);
	}

	TFTensor* AllocateWithData(const vector<int>& shape, const vector<uint>& data, const TFType type = TFType::Float) {
		TFTensor* tensor_memory = Allocate(shape, type);
		SetDataAtOffset(tensor_memory, 0, data);
		return tensor_memory;
	}

	virtual vector<uint> Readback(const TFTensor* memory) = 0;
	virtual uint ReadbackValue(const TFTensor* memory, uint index) = 0;
	virtual void Writeback(const TFTensor* memory, const vector<uint>& data) = 0;
	virtual void WritebackValue(const TFTensor* memory, uint index, uint value) = 0;

	TFTensor* MakeTensor(uint* shape, uint dim, TFBuffer* buf, TFType type) {
		TFTensor* tensor = new TFTensor();
		tensor->buffer = buf;
		tensor->dim = dim;
		tensor->shape = shape;
		tensor->type = type;
		return tensor;
	}

	TFTensor* MakeTensor(const vector<int>& shape, TFBuffer* buf, TFType type) {
		uint* shape_arr = new uint[shape.size()];
		std::copy(shape.begin(), shape.end(), shape_arr);
		return MakeTensor(shape_arr, (int)shape.size(), buf, type);
	}

	void Free(TFTensor* memory) {
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