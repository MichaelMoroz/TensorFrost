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
	unordered_map<Buffer*, uint*> allocated_arrays;

	Buffer* TryGetBuffer(int size) override {
		Buffer* buffer = buffer_manager.TryAllocateBuffer(size);
		if(!allocated_arrays.contains(buffer)) {
			allocated_arrays[buffer] = CreateBuffer(size);
		}
		return buffer;
	}

	uint* GetNativeBuffer(const TensorProp* mem) {
		if(!allocated_arrays.contains(mem->buffer)) {
			throw std::runtime_error("Tensor memory not allocated");
		}
		return allocated_arrays[mem->buffer];
	}

	void SetDataAtOffset(const TensorProp* buffer, int offset, const vector<uint>& data) override {
		uint* array = allocated_arrays[buffer->buffer];
		memcpy(array + offset, data.data(), data.size() * sizeof(uint));
	}

	uint* CreateBuffer(int size) {
		return new uint[size];
	}

	vector<uint> Readback(const TensorProp* mem) override {
		uint* array = GetNativeBuffer(mem);
		vector<uint> data(mem->buffer->size);
		for(int i = 0; i < mem->buffer->size; i++) {
			data[i] = array[i];
		}
		return data;
	}

	uint ReadbackValue(const TensorProp* mem, uint index) override {
		uint* array = GetNativeBuffer(mem);
		return array[index];
	}

	void Writeback(const TensorProp* mem, const vector<uint>& data) override {
		uint* array = GetNativeBuffer(mem);
		memcpy(array, data.data(), data.size() * sizeof(uint));
	}

	void WritebackValue(const TensorProp* mem, uint index, uint value) override {
		uint* array = GetNativeBuffer(mem);
		array[index] = value;
	}

	~CpuMemoryManager() {
		for(auto& [buffer, array]: allocated_arrays) {
			delete[] array;
		}
	}
};

}  // namespace TensorFrost