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
	 const int DEFAULT_SIZE = 1024 * 1024 * 64;
	 GLuint memory;
	 int mem_size;

	 map<int, GLuint> temporary_buffers;

	 OpenGLMemoryManager() {
		 mem_size = DEFAULT_SIZE;
		 memory = CreateBuffer(DEFAULT_SIZE);
	 }

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

	 void IncreaseMemorySize(int new_size) {
		 GLuint new_memory = CreateBuffer(new_size);
		 CopyBuffer(memory, new_memory, mem_size);
		 DeleteBuffer(memory);
		 memory = new_memory;
		 mem_size = new_size;

		 //throw std::runtime_error("Memory size exceeded, increased to " + std::to_string(new_size));
	 }

	 void SetDataAtOffset(uint offset, const std::vector<uint>& data) {
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory);
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

	 TensorMemory* Allocate(const vector<int>& shape, const DataType type = DataType::Float) override {
		 int size = GetLinearSize(shape);

		 if (size == 0) {
			 throw invalid_argument("Trying to allocate a tensor with size 0");
		 }

		 Frame* frame = allocator.AllocateFrame(size);
		 // reserve space in memory if needed
		 if ((int)frame->end > mem_size) {
			 IncreaseMemorySize(frame->end * 3 / 2);
		 }

		 auto* tensor_memory = new TensorMemory(shape, frame, this);
		 tensor_memory->type = type;
		 allocated_by_offset[frame->start] = tensor_memory;
		 return tensor_memory;
	 }

	 TensorMemory* AllocateWithData(const vector<int>& shape, const vector<uint>& data, const DataType type = DataType::Float) override {
		 TensorMemory* tensor_memory = Allocate(shape, type);
		 SetDataAtOffset(tensor_memory->frame->start, data);
		 return tensor_memory;
	 }

	 GLuint TryGetTemporaryBuffer(int size) {
		 //find the smallest buffer that can fit the size, which isnt larger than 2x the size
		 auto it = temporary_buffers.lower_bound(size);
		 if (it != temporary_buffers.end() && it->first <= size * 2) {
			return it->second;
		 } 

		 GLuint buffer = CreateBuffer(size);
		 temporary_buffers[size] = buffer;
		 return buffer;
	 }
	 
	 void ReadbackUsingBuffer(uint memory_offset, uint size, uint* buffer) {
		 // create a new ssbo with the same size as the tensor
		 GLuint temp_memory = TryGetTemporaryBuffer(size);
		 CopyBuffer(memory, temp_memory, size, memory_offset);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, temp_memory);
		 glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size * sizeof(uint), buffer);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	 }

	 uint ReadbackValue(const TensorMemory* mem, uint index) {
		 uint data;
		 ReadbackUsingBuffer(mem->frame->start + index, 1, &data);
		 return data;
	 }

	 vector<uint> Readback(const TensorMemory* mem) override {
		  vector<uint> data;
		  data.resize(mem->GetSize());
		  ReadbackUsingBuffer(mem->frame->start, mem->GetSize(), data.data());
		  return data;
	  }

	 void WritebackValue(const TensorMemory* mem, uint index, uint value) {
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory);
		 glBufferSubData(GL_SHADER_STORAGE_BUFFER,
		                 (mem->frame->start + index) * sizeof(uint), sizeof(uint),
		                 &value);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	 }

	 void Writeback(const TensorMemory* mem, const vector<uint>& data) override {
		 SetDataAtOffset(mem->frame->start, data);
		 glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	 }


	 ~OpenGLMemoryManager() {
		 DeleteBuffer(memory);
		 for (auto& [size, buffer] : temporary_buffers) {
			 DeleteBuffer(buffer);
		 }
	 }

	 uint32_t GetAllocatedSize() const {
		return mem_size;
	}
};


}  // namespace TensorFrost