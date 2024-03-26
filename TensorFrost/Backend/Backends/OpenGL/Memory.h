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
	 const int DEFAULT_SIZE = 1024 * 1024 * 8;
	 GLuint memory;
	 int mem_size;

	 OpenGLMemoryManager() {
		 mem_size = DEFAULT_SIZE;
		 memory = CreateBuffer(DEFAULT_SIZE);
	 }

	 GLuint CreateBuffer(int size) {
		 GLuint buffer;
		 glGenBuffers(1, &buffer);
		 CheckError("glGenBuffers");

		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
		 CheckError("glBindBuffer");

		 glBufferData(GL_SHADER_STORAGE_BUFFER, size * sizeof(uint), nullptr, GL_DYNAMIC_COPY);
		 CheckError("glBufferData");

		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 // No need to check error after unbinding

		 return buffer;
	 }

	 void CopyBuffer(GLuint source, GLuint dest, int size) {
		 glBindBuffer(GL_COPY_READ_BUFFER, source);
		 CheckError("glBindBuffer - GL_COPY_READ_BUFFER");

		 glBindBuffer(GL_COPY_WRITE_BUFFER, dest);
		 CheckError("glBindBuffer - GL_COPY_WRITE_BUFFER");

		 glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, size * sizeof(uint));
		 CheckError("glCopyBufferSubData");

		 glBindBuffer(GL_COPY_READ_BUFFER, 0);
		 glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
		 // No need to check error after unbinding
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
	 }

	 void SetDataAtOffset(uint offset, const std::vector<uint>& data) {
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory);
		 CheckError("glBindBuffer - GL_SHADER_STORAGE_BUFFER for SetDataAtOffset");

		 glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset * sizeof(uint),
		                 data.size() * sizeof(uint), data.data());
		 CheckError("glBufferSubData");

		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 // No need to check error after unbinding
	 }

	 void CheckError(const char* operation) {
		 #ifdef NDEBUG
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

	  uint ReadbackValue(const TensorMemory* mem, uint index) {
		 uint data;
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory);
		 glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,
		                    (mem->frame->start + index) * sizeof(uint),
		                    sizeof(uint), &data);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 return data;
	 }

	 void WritebackValue(const TensorMemory* mem, uint index, uint value) {
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory);
		 glBufferSubData(GL_SHADER_STORAGE_BUFFER,
		                 (mem->frame->start + index) * sizeof(uint), sizeof(uint),
		                 &value);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	 }

	 vector<uint> Readback(const TensorMemory* mem) override {
		 vector<uint> data;
		 data.resize(mem->GetSize());
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory);
		 glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, mem->frame->start * sizeof(uint),
		                    data.size() * sizeof(uint), data.data());
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 return data;
	 }

	 void Writeback(const TensorMemory* mem, const vector<uint>& data) override {
		 SetDataAtOffset(mem->frame->start, data);
	 }


	 ~OpenGLMemoryManager() {
		 DeleteBuffer(memory);
	 }
};


}  // namespace TensorFrost