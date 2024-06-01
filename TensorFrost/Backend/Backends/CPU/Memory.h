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
	unordered_map<TFBuffer*, uint32_t*> allocated_arrays;

	void CleanUp() {
		for(auto buf_to_delete: buffer_manager.buffers_to_delete) {
			DeleteBuffer(allocated_arrays[buf_to_delete]);
			allocated_arrays.erase(buf_to_delete);
			buffer_manager.RemoveBuffer(buf_to_delete);
		}
		buffer_manager.buffers_to_delete.clear();
	}

	TFBuffer* TryGetBuffer(size_t size) override {
		buffer_manager.UpdateTick();
		CleanUp();

		TFBuffer* buffer = buffer_manager.TryAllocateBuffer(size);
		if(!allocated_arrays.contains(buffer)) {
			allocated_arrays[buffer] = CreateBuffer(buffer->size);
		}
		return buffer;
	}

	uint* GetNativeBuffer(const TFTensor* mem) {
		if(!allocated_arrays.contains(mem->buffer)) {
			throw std::runtime_error("Tensor memory not allocated");
		}
		return allocated_arrays[mem->buffer];
	}

	void SetDataAtOffset(const TFTensor* buffer, size_t offset, const vector<uint32_t>& data) override {
		uint32_t* array = allocated_arrays[buffer->buffer];
		memcpy(array + offset, data.data(), data.size() * sizeof(uint32_t));
	}

	uint32_t* CreateBuffer(size_t size) {
		return new uint32_t[size];
	}

	vector<uint> Readback(const TFTensor* mem) override {
		uint32_t* array = GetNativeBuffer(mem);
		vector<uint> data(mem->buffer->size);
		for(int i = 0; i < mem->buffer->size; i++) {
			data[i] = array[i];
		}
		return data;
	}

	uint ReadbackValue(const TFTensor* mem, size_t index) override {
		uint32_t* array = GetNativeBuffer(mem);
		return array[index];
	}

	void Writeback(const TFTensor* mem, const vector<uint32_t>& data) override {
		uint32_t* array = GetNativeBuffer(mem);
		memcpy(array, data.data(), data.size() * sizeof(uint));
	}

	void WritebackValue(const TFTensor* mem, size_t index, uint32_t value) override {
		uint32_t* array = GetNativeBuffer(mem);
		array[index] = value;
	}

	void DeleteBuffer(uint* buffer) {
		delete[] buffer;
	}

	~CpuMemoryManager() {
		for(auto& [buffer, array]: allocated_arrays) {
			DeleteBuffer(array);
		}
	}
};

}  // namespace TensorFrost