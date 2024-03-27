#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstring>

#include "../../TensorMemory.h"

namespace TensorFrost {

using namespace std;

class CpuMemoryManager : public TensorMemoryManager {
 public:
	vector<uint> memory;

	TensorMemory* Allocate(const vector<int>& shape,
	                       const DataType type = DataType::Float) override {
		int size = GetLinearSize(shape);

		if (size == 0) {
			throw invalid_argument("Trying to allocate a tensor with size 0");
		}

		Frame* frame = allocator.AllocateFrame(size);
		// reserve space in memory if needed
		if (frame->end > memory.size()) {
			memory.resize(frame->end * 3 / 2);
		}

		auto* tensor_memory = new TensorMemory(shape, frame, this);
		tensor_memory->type = type;
		allocated_by_offset[frame->start] = tensor_memory;
		return tensor_memory;
	}

	TensorMemory* AllocateWithData(const vector<int>& shape,
	                               const vector<uint>& data,
	    const DataType type = DataType::Float) override {
		TensorMemory* tensor_memory = Allocate(shape, type);
		memcpy(memory.data() + tensor_memory->frame->start, data.data(),
		       data.size() * sizeof(uint));
		return tensor_memory;
	}

	vector<uint> Readback(const TensorMemory* mem) override {
		vector<uint> data;
		data.resize(mem->GetSize());
		memcpy(data.data(), this->memory.data() + mem->frame->start,
		       data.size() * sizeof(uint));
		return data;
	}

	uint ReadbackValue(const TensorMemory* mem, uint index) override {
		return memory[mem->frame->start + index];
	}

	void Writeback(const TensorMemory* mem, const vector<uint>& data) override {
		memcpy(memory.data() + mem->frame->start, data.data(),
					       data.size() * sizeof(uint));
	}

	void WritebackValue(const TensorMemory* mem, uint index, uint value) override {
		memory[mem->frame->start + index] = value;
	}

	uint32_t GetAllocatedSize() const {
		return memory.capacity();
	}
};

}  // namespace TensorFrost