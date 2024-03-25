#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../Tensor/Tensor.h"
#include "FrameAllocator.h"

namespace TensorFrost {

using namespace std;

extern "C" {
	struct TensorProp {
		uint offset;
		uint dim;
		uint* shape;
		DataType type;
	};

	typedef TensorProp alloc_func(uint*&, uint*, uint, DataType);
	typedef void dealloc_func(TensorProp);
	typedef uint readback_func(TensorProp, uint);
	typedef void writeback_func(TensorProp, uint, uint);
	typedef void dispatch_func(int, TensorProp*, uint*, uint*);
}

using uint = unsigned int;
using main_func = void(TensorProp*, TensorProp*, alloc_func, dealloc_func, readback_func, writeback_func, dispatch_func);

int GetLinearSize(const vector<int>& shape);

class TensorMemory;

class TensorMemoryManager {
 public:
	FrameAllocator allocator;
	map<uint, TensorMemory*> allocated_by_offset;

	virtual TensorMemory* Allocate(const vector<int>& shape,
	                               const DataType type = DataType::Float) = 0;
	virtual TensorMemory* AllocateWithData(const vector<int>& shape, const vector<uint>& data,
	    const DataType type = DataType::Float) = 0;
	virtual vector<uint> Readback(const TensorMemory* memory) = 0;
	virtual uint ReadbackValue(const TensorMemory* memory, uint index) = 0;
	virtual void Writeback(const TensorMemory* memory, const vector<uint>& data) = 0;
	virtual void WritebackValue(const TensorMemory* memory, uint index, uint value) = 0;
	virtual void Free(TensorMemory* memory) = 0;
	virtual void Free(uint offset) = 0;

	uint32_t GetAllocatedSize() const {
		return allocator.GetRequiredAllocatedStorage();
	}

	virtual ~TensorMemoryManager() = default;
};

class TensorMemory {
 public:
	DataType type = DataType::Float;
	vector<int> shape;
	TensorMemoryManager* manager;
	Frame* frame;

	TensorMemory(const vector<int>& shape, Frame* frame,
	             TensorMemoryManager* used_manager)
	    : shape(shape), frame(frame), manager(used_manager) {}

	int GetSize() const { return GetLinearSize(shape); }

	vector<int> GetShape() const { return shape; }

	~TensorMemory() { manager->Free(this); }
};

}  // namespace TensorFrost