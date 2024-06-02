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

class TFOpenGLBuffer: public TFBuffer {
 public:
	GLuint buffer;

	static GLuint CreateBuffer(size_t size) {
		GLint maxsize;
		glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxsize);

		if (size * sizeof(uint) > maxsize) {
			throw std::runtime_error("SSBO memory size exceeded, max size is " + std::to_string(maxsize));
		}

		GLuint buffer;
		glCreateBuffers(1, &buffer);
		glNamedBufferStorage(buffer, size * sizeof(uint32_t), nullptr, GL_DYNAMIC_STORAGE_BIT);
		return buffer;
	}

	static void DeleteBuffer(GLuint buffer) {
		glDeleteBuffers(1, &buffer);
	}

	TFOpenGLBuffer(size_t size): TFBuffer(size) {
		buffer = CreateBuffer(size);
	}

	~TFOpenGLBuffer() {
		DeleteBuffer(buffer);
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

	 GLuint GetNativeBuffer(const TFTensor* mem) {
		 return static_cast<TFOpenGLBuffer*>(mem->buffer)->buffer;
	 }

	 void CopyBuffer(GLuint source, GLuint dest, size_t size, size_t read_offset = 0, size_t write_offset = 0) {
		 glBindBuffer(GL_COPY_READ_BUFFER, source);
		 glBindBuffer(GL_COPY_WRITE_BUFFER, dest);
		 glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, read_offset * sizeof(uint), write_offset * sizeof(uint), size * sizeof(uint));
		 glBindBuffer(GL_COPY_READ_BUFFER, 0);
		 glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
	 }

	 void SetDataAtOffset(const TFTensor* buffer, size_t offset, const vector<uint32_t>& data) override {
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, GetNativeBuffer(buffer));
		 CheckError("glBindBuffer - GL_SHADER_STORAGE_BUFFER for SetDataAtOffset");

		 glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset * sizeof(uint32_t),
		                 data.size() * sizeof(uint32_t), data.data());
		 CheckError("glBufferSubData");

		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	 }

	 void CheckError(const char* operation) {
		 #ifndef NDEBUG
		 GLenum error = glGetError();
		 if (error != GL_NO_ERROR) {
			throw std::runtime_error(std::string(operation) + " failed with error code " + std::to_string(error));
		 }
		 #endif
	 }
	 
	 void ReadbackBuffer(const TFTensor* mem, size_t offset, size_t size, uint32_t* buffer) {
		 // create a new ssbo with the same size as the tensor
		 GLuint memory = GetNativeBuffer(mem);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory);
		 glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, offset * sizeof(uint32_t), size * sizeof(uint32_t), buffer);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	 }

	 uint ReadbackValue(const TFTensor* mem, size_t index) {
		 uint data;
		 ReadbackBuffer(mem, index, 1, &data);
		 return data;
	 }

	 vector<uint32_t> Readback(const TFTensor* mem) override {
		  vector<uint32_t> data;
		  data.resize(GetSize(mem));
		  ReadbackBuffer(mem, 0, GetSize(mem), data.data());
		  return data;
	  }

	 void WritebackValue(const TFTensor* mem, size_t index, uint32_t value) {
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, GetNativeBuffer(mem));
		 glBufferSubData(GL_SHADER_STORAGE_BUFFER,
		                 (index) * sizeof(uint32_t), sizeof(uint32_t),
		                 &value);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	 }

	 void Writeback(const TFTensor* mem, const vector<uint32_t>& data) override {
		 SetDataAtOffset(mem, 0, data);
		 glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	 }
};


}  // namespace TensorFrost