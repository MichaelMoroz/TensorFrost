#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../Tensor/Tensor.h"

namespace TensorFrost {

using namespace std;

extern "C" {
	struct TFBuffer {
		size_t size = 0;
		bool up_to_date = true;
		bool read_only = false;
		//add type descriptor (for special kinds of buffers)
	};

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

	struct TFRuntime {
		alloc_func* alloc;
		dealloc_func* dealloc;
		readback_func* readback;
		writeback_func* writeback;
		dispatch_func* dispatch;
		void* custom_data;
	};

	typedef void cpu_dispatch_func(const uint32_t* var, uint32_t** mem, uint work_group_count);
	typedef void main_func(TFTensor*, TFTensor*, TFRuntime);
}

class TFBufferTemplate : public TFBuffer {
public:
	TFBufferTemplate(size_t size) : TFBuffer(size) {}

	virtual void SetDataAtOffset(size_t offset, const vector<uint32_t>& data) {
		throw std::runtime_error("SetDataAtOffset not implemented");
	}
	virtual void GetDataAtOffset(size_t offset, size_t size, uint32_t* data) {
		throw std::runtime_error("GetDataAtOffset not implemented");
	}
};

using uint = unsigned int;

size_t GetLinearSize(const vector<size_t>& shape);
vector<size_t> GetShape(const TFTensor* tensor);
size_t GetSize(const TFTensor* tensor);

class TensorMemoryManager {
private:
	const int MAX_UNUSED_TIME = 512;
	map<size_t, unordered_set<TFBuffer*>> allocated_buffers;
	map<TFBuffer*, int> unused_time;
	unordered_set<TFBuffer*> used_buffers;

	static TFTensor* MakeTensor(size_t* shape, size_t dim, TFBuffer* buf, TFType type);
	static TFTensor* MakeTensor(const vector<size_t>& shape, TFBuffer* buf, TFType type);
	void UpdateTick();

	TFBuffer* AllocateBuffer(size_t size);
	TFBuffer* TryAllocateBuffer(size_t size);
	void DeallocateBuffer(TFBuffer* buffer);
	void RemoveBuffer(TFBuffer* buffer);

protected:
	virtual TFBuffer* CreateBuffer(size_t size) {
		throw std::runtime_error("CreateBuffer not implemented");
	}

	virtual void DeleteBuffer(TFBuffer * buffer) {
		throw std::runtime_error("DeleteBuffer not implemented");
	}

public:
	virtual vector<uint32_t> Readback(const TFTensor* memory);
	virtual uint ReadbackValue(const TFTensor* memory, size_t index);
	virtual void Writeback(const TFTensor* memory, const vector<uint32_t>& data);
	virtual void WritebackValue(const TFTensor* memory, size_t index, uint32_t value);

	TFTensor* AllocateTensor(const vector<size_t>& shape, const TFType type = TFType::Float);
	TFTensor* AllocateTensorWithData(const vector<size_t>& shape, const vector<uint32_t>& data, const TFType type = TFType::Float, bool read_only = false);
	void DeallocateTensor(TFTensor tensor);

	size_t GetAllocatedSize() const;
	size_t GetUnusedAllocatedSize() const;

	~TensorMemoryManager();
};


extern TensorMemoryManager* global_memory_manager;

}  // namespace TensorFrost