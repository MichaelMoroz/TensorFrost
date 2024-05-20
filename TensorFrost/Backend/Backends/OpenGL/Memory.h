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
	 const int MAX_BUFFER_SIZE = 2147483647 / sizeof(uint);

	 unordered_map<int, unordered_set<GLuint>> allocated_buffers;
	 unordered_map<const TensorMemory*, GLuint> tensor_buffers;
	 unordered_set<GLuint> used_buffers;

	 OpenGLMemoryManager() {}

	 GLuint AllocateBuffer(int size) {
		GLuint buffer = CreateBuffer(size);
	 	//add the buffer to the list of allocated buffers
	 	allocated_buffers[size].insert(buffer);
	 	return buffer;
	 }

	 void DeallocateBuffer(GLuint buffer) {
		 used_buffers.erase(buffer);
	 }

	 void RemoveBuffer(GLuint buffer) {
		 for(auto& [size, buffers]: allocated_buffers) {
			 buffers.erase(buffer);
		 }
		 DeallocateBuffer(buffer);
		 DeleteBuffer(buffer);
	 }

	 GLuint GetBufferAtOffset(uint offset) {
	 	 if(!allocated_by_offset.contains(offset)) {
	 		 throw std::runtime_error("No tensor allocated at offset " + std::to_string(offset));
	 	 }
		 return tensor_buffers[allocated_by_offset[offset]];
	 }

	 GLuint AllocateTensor(int size) {
		 //try to find a non-used buffer of the correct size
	 	 GLuint buffer = 0;
	 	 bool found = false;
	 	 for(auto ssbo: allocated_buffers[size]) {
	 		 if(used_buffers.contains(ssbo)) {
	 			 continue;
	 		 }
	 	 	 buffer = ssbo;
	 	 	 found = true;
	 	 }
	 	 //if no buffer was found, create a new one
	 	 if(!found) {
	 		 buffer = AllocateBuffer(size);
	 	 }
	 	 used_buffers.insert(buffer);
	 	 return buffer;
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

	 void SetDataAtOffset(GLuint memory, uint offset, const std::vector<uint>& data) {
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

		 GLuint memory = AllocateTensor(size);

		 auto* tensor_memory = new TensorMemory(shape, frame, this);
	 	 tensor_buffers[tensor_memory] = memory;
		 tensor_memory->type = type;
		 allocated_by_offset[frame->start] = tensor_memory;
		 return tensor_memory;
	 }

	 TensorMemory* AllocateWithData(const vector<int>& shape, const vector<uint>& data, const DataType type = DataType::Float) override {
		 TensorMemory* tensor_memory = Allocate(shape, type);
		 SetDataAtOffset(tensor_buffers[tensor_memory], 0, data);
		 return tensor_memory;
	 }
	 
	 void ReadbackBuffer(const TensorMemory* mem, uint offset, uint size, uint* buffer) {
		 // create a new ssbo with the same size as the tensor
		 GLuint memory = GetBufferAtOffset(mem->frame->start);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory);
		 glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, offset * sizeof(uint), size * sizeof(uint), buffer);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	 }

	 uint ReadbackValue(const TensorMemory* mem, uint index) {
		 uint data;
		 ReadbackBuffer(mem, index, 1, &data);
		 return data;
	 }

	 vector<uint> Readback(const TensorMemory* mem) override {
		  vector<uint> data;
		  data.resize(mem->GetSize());
		  ReadbackBuffer(mem, 0, mem->GetSize(), data.data());
		  return data;
	  }

	 void FreeBuff(TensorMemory *memory) override {
		 DeallocateBuffer(GetBufferAtOffset(memory->frame->start));
	 }

	 void WritebackValue(const TensorMemory* mem, uint index, uint value) {
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, tensor_buffers[mem]);
		 glBufferSubData(GL_SHADER_STORAGE_BUFFER,
		                 (index) * sizeof(uint), sizeof(uint),
		                 &value);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	 }

	 void Writeback(const TensorMemory* mem, const vector<uint>& data) override {
		 SetDataAtOffset(tensor_buffers[mem], 0, data);
		 glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	 }


	 ~OpenGLMemoryManager() {
	 	 for(auto& [mem, buffer]: tensor_buffers) {
			 DeallocateBuffer(buffer);
		 }
		 for(auto& [size, buffers]: allocated_buffers) {
			 for(auto buffer: buffers) {
				 DeleteBuffer(buffer);
			 }
		 }
	 }

	 uint32_t GetAllocatedSize() const {
		uint32_t allocated = 0;
	 	for(auto& [size, buffers]: allocated_buffers) {
	 		allocated += (uint32_t)size * (uint32_t)buffers.size();
	 	}
	 	return allocated;
	 }
};


}  // namespace TensorFrost