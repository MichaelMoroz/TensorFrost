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
		TFType type;
		size_t dim;
		const size_t* shape;
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

	typedef TFTensor alloc_func(const size_t*, size_t, TFType, void*);
	typedef void dealloc_func(TFTensor, void*);
	typedef uint readback_func(TFTensor, size_t, void*);
	typedef void writeback_func(TFTensor, size_t, uint32_t, void*);
	typedef void dispatch_func(TFDispatchInfo, void*);
	typedef void cpu_dispatch_func(const uint32_t* var, uint32_t** mem, uint work_group_count);

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

size_t GetLinearSize(const vector<size_t>& shape);
vector<size_t> GetShape(const TFTensor* tensor);
size_t GetSize(const TFTensor* tensor);

class TensorMemoryManager;

class TensorMemoryManager {
public:
	BufferManager buffer_manager;

	virtual TFBuffer* TryGetBuffer(size_t size) {
		throw std::runtime_error("TryGetBuffer not implemented");
	}

	virtual void SetDataAtOffset(const TFTensor* buffer, size_t offset, const vector<uint32_t>& data) {
		throw std::runtime_error("SetDataAtOffset not implemented");
	}

	TFTensor* Allocate(const vector<size_t>& shape, const TFType type = TFType::Float, bool read_only = false) {
		size_t size = GetLinearSize(shape);

		if (size == 0) {
			throw invalid_argument("Trying to allocate a tensor with size 0");
		}

		TFBuffer* buf = TryGetBuffer(size);
		buf->read_only = read_only;
		return MakeTensor(shape, buf, type);
	}

	TFTensor* AllocateWithData(const vector<size_t>& shape, const vector<uint32_t>& data, const TFType type = TFType::Float, bool read_only = false) {
		TFTensor* tensor_memory = Allocate(shape, type);
		SetDataAtOffset(tensor_memory, 0, data);
		return tensor_memory;
	}

	virtual vector<uint32_t> Readback(const TFTensor* memory) = 0;
	virtual uint ReadbackValue(const TFTensor* memory, size_t index) = 0;
	virtual void Writeback(const TFTensor* memory, const vector<uint32_t>& data) = 0;
	virtual void WritebackValue(const TFTensor* memory, size_t index, uint32_t value) = 0;

	TFTensor* MakeTensor(size_t* shape, size_t dim, TFBuffer* buf, TFType type) {
		TFTensor* tensor = new TFTensor();
		tensor->buffer = buf;
		tensor->dim = dim;
		tensor->shape = shape;
		tensor->type = type;
		return tensor;
	}

	TFTensor* MakeTensor(const vector<size_t>& shape, TFBuffer* buf, TFType type) {
		size_t* shape_arr = new size_t[shape.size()];
		std::copy(shape.begin(), shape.end(), shape_arr);
		return MakeTensor(shape_arr, (int)shape.size(), buf, type);
	}

	void Free(TFTensor* memory) {
		buffer_manager.DeallocateBuffer(memory->buffer);
	}

	size_t GetAllocatedSize() const {
		return buffer_manager.GetRequiredAllocatedStorage();
	}

	size_t GetUnusedAllocatedSize() const {
		return buffer_manager.GetUnusedAllocatedStorage();
	}
};


extern TensorMemoryManager* global_memory_manager;

}  // namespace TensorFrost