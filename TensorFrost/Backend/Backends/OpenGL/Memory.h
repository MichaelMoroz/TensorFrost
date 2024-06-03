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

class TFOpenGLBuffer: public TFBufferTemplate {
	GLuint buffer;

	const size_t max_cache_size = 16384;
	uint32_t* cached_data = nullptr;

 public:
	TFOpenGLBuffer(size_t size): TFBufferTemplate(size) {
		GLint maxsize;
		glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxsize);

		if (size * sizeof(uint32_t) > maxsize) {
			throw std::runtime_error("SSBO memory size exceeded, max size is " + std::to_string(maxsize));
		}

		glCreateBuffers(1, &buffer);
		glNamedBufferStorage(buffer, size * sizeof(uint32_t), nullptr, GL_DYNAMIC_STORAGE_BIT);
	}

	void UpdateCache(size_t data_offset, size_t data_size, const uint32_t* data) {
		if(data_offset == 0 && data_size == used_size && data_size <= max_cache_size) {
			if(cached_data == nullptr) {
				cached_data = new uint32_t[data_size];
			}
			memcpy(cached_data, data, data_size * sizeof(uint32_t));
			up_to_date = true;
		}
	}

	void SetDataAtOffset(size_t offset, const vector<uint32_t>& data) override {
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset * sizeof(uint32_t),
						data.size() * sizeof(uint32_t), data.data());
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		UpdateCache(offset, data.size(), data.data());
	}

	void GetDataAtOffset(size_t offset, size_t size, uint32_t* data) override {
		if(size <= max_cache_size && up_to_date) {
			memcpy(data, cached_data + offset, size * sizeof(uint32_t));
			return;
		}

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
		glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, offset * sizeof(uint32_t),
						   size * sizeof(uint32_t), data);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		UpdateCache(offset, size, data);
	}

	GLuint GetNative() const {
		return buffer;
	}

	~TFOpenGLBuffer() {
		glDeleteBuffers(1, &buffer);
	}
};

class OpenGLMemoryManager : public TensorMemoryManager {
 public:
	 OpenGLMemoryManager() {}

	 TFBuffer* CreateBuffer(size_t size) override {
	 	return new TFOpenGLBuffer(size);
	 }

	 void DeleteBuffer(TFBuffer* buffer) override {
	 	delete (TFOpenGLBuffer*)buffer;
	 }
};


}  // namespace TensorFrost