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

	TensorMemory* Allocate(const vector<int>& shape) override {
		int size = GetLinearSize(shape);
		Frame* frame = allocator.AllocateFrame(size);
		// reserve space in memory if needed
		if (frame->end > memory.size()) {
			memory.resize(frame->end * 3 / 2);
		}

		auto* tensor_memory = new TensorMemory(shape, frame, this);
		allocated_by_offset[frame->start] = tensor_memory;
		return tensor_memory;
	}

	TensorMemory* AllocateWithData(const vector<int>& shape,
	                               const vector<uint>& data) override {
		TensorMemory* tensor_memory = Allocate(shape);
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

	void Free(TensorMemory* memory) override {
		Frame* frame = memory->frame;
		allocator.FreeFrame(*frame);
		allocated_by_offset.erase(frame->start);
	}

	void Free(uint offset) override { 
		Free(allocated_by_offset[offset]);
	}

	~CpuMemoryManager() override {
		for (auto& pair : allocated_by_offset) {
			delete pair.second;
		}
	}
};

}  // namespace TensorFrost