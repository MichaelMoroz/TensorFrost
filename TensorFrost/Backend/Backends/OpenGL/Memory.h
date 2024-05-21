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

class OpenGLMemoryManager : public TensorMemoryManager {
 public:
	 unordered_map<Buffer*, GLuint> allocated_ssbo;

	 OpenGLMemoryManager() {}

	 GLuint CreateBuffer(int size) {
		 GLint maxsize;
		 glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxsize);

		 if (size * sizeof(uint) > maxsize) {
			 throw std::runtime_error("SSBO memory size exceeded, max size is " + std::to_string(maxsize));
		 }

		 GLuint buffer;
		 glCreateBuffers(1, &buffer);
		 glNamedBufferStorage(buffer, size * sizeof(uint), nullptr, GL_DYNAMIC_STORAGE_BIT);
		 return buffer;
	 }

	 Buffer* TryGetBuffer(int size) override {
	 	Buffer* buffer = buffer_manager.TryAllocateBuffer(size);
	 	if(!allocated_ssbo.contains(buffer)) {
	 		allocated_ssbo[buffer] = CreateBuffer(size);
	 	}
	 	return buffer;
	 }

	 GLuint GetNativeBuffer(const TensorProp* mem) {
		 if(!allocated_ssbo.contains(mem->buffer)) {
			 throw std::runtime_error("Tensor memory not allocated");
		 }
	 	 return allocated_ssbo[mem->buffer];
	 }

	 void CopyBuffer(GLuint source, GLuint dest, int size, int read_offset = 0, int write_offset = 0) {
		 glBindBuffer(GL_COPY_READ_BUFFER, source);
		 glBindBuffer(GL_COPY_WRITE_BUFFER, dest);
		 glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, read_offset * sizeof(uint), write_offset * sizeof(uint), size * sizeof(uint));
		 glBindBuffer(GL_COPY_READ_BUFFER, 0);
		 glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
	 }

	 void DeleteBuffer(GLuint buffer) {
		 glDeleteBuffers(1, &buffer);
		 CheckError("glDeleteBuffers");
	 }

	 void SetDataAtOffset(const TensorProp* buffer, int offset, const vector<uint>& data) override {
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, GetNativeBuffer(buffer));
		 CheckError("glBindBuffer - GL_SHADER_STORAGE_BUFFER for SetDataAtOffset");

		 glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset * sizeof(uint),
		                 data.size() * sizeof(uint), data.data());
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
	 
	 void ReadbackBuffer(const TensorProp* mem, uint offset, uint size, uint* buffer) {
		 // create a new ssbo with the same size as the tensor
		 GLuint memory = GetNativeBuffer(mem);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory);
		 glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, offset * sizeof(uint), size * sizeof(uint), buffer);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	 }

	 uint ReadbackValue(const TensorProp* mem, uint index) {
		 uint data;
		 ReadbackBuffer(mem, index, 1, &data);
		 return data;
	 }

	 vector<uint> Readback(const TensorProp* mem) override {
		  vector<uint> data;
		  data.resize(GetSize(mem));
		  ReadbackBuffer(mem, 0, GetSize(mem), data.data());
		  return data;
	  }

	 void WritebackValue(const TensorProp* mem, uint index, uint value) {
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, GetNativeBuffer(mem));
		 glBufferSubData(GL_SHADER_STORAGE_BUFFER,
		                 (index) * sizeof(uint), sizeof(uint),
		                 &value);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	 }

	 void Writeback(const TensorProp* mem, const vector<uint>& data) override {
		 SetDataAtOffset(mem, 0, data);
		 glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	 }


	 ~OpenGLMemoryManager() {
		 for(auto& [buf, buffer]: allocated_ssbo) {
			DeleteBuffer(buffer);
		 }
	 }
};


}  // namespace TensorFrost