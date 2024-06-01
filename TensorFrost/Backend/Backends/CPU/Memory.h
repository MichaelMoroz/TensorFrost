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

class TFCPUBuffer: public TFBuffer {
public:
	uint32_t* data;

	TFCPUBuffer(size_t size): TFBuffer(size) {
		data = new uint32_t[size];
	}

	~TFCPUBuffer() {
		delete[] data;
	}
};

class CpuMemoryManager : public TensorMemoryManager {
 public:
	TFBuffer* CreateBuffer(size_t size) override {
		return new TFCPUBuffer(size);
	}

	uint* GetNativeBuffer(const TFTensor* mem) {
		return static_cast<TFCPUBuffer*>(mem->buffer)->data;
	}

	void SetDataAtOffset(const TFTensor* buffer, size_t offset, const vector<uint32_t>& data) override {
		uint32_t* array = GetNativeBuffer(buffer);
		memcpy(array + offset, data.data(), data.size() * sizeof(uint32_t));
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
};

}  // namespace TensorFrost