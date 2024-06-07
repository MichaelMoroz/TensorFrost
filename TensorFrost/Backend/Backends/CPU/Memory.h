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

class TFCPUBuffer: public TFBufferTemplate {
public:
	uint32_t* data;

	TFCPUBuffer(size_t size): TFBufferTemplate(size) {
		data = new uint32_t[size];
	}

	void UpdateName(const char* new_name) override {
		if(new_name != nullptr) {
			name = new_name;
		}
	}

	void SetDataAtOffset(size_t offset, const vector<uint32_t>& data) override {
		memcpy(this->data + offset, data.data(), data.size() * sizeof(uint32_t));
	}

	void GetDataAtOffset(size_t offset, size_t size, uint32_t* data) override {
		memcpy(data, this->data + offset, size * sizeof(uint32_t));
	}

	uint32_t* GetNative() const {
		return data;
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

	void DeleteBuffer(TFBuffer* buffer) override {
		delete (TFCPUBuffer*)buffer;
	}
};

}  // namespace TensorFrost