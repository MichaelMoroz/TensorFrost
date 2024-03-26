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
	 int size;

	 OpenGLMemoryManager() {
		 this->size = DEFAULT_SIZE;
		 memory = CreateBuffer(DEFAULT_SIZE);
	 }

	 GLuint CreateBuffer(int size) {
		 GLuint buffer;
		 glGenBuffers(1, &buffer);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
		 glBufferData(GL_SHADER_STORAGE_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 return buffer;
	 }

	 void CopyBuffer(GLuint source, GLuint dest, int size) {
		 glBindBuffer(GL_COPY_READ_BUFFER, source);
		 glBindBuffer(GL_COPY_WRITE_BUFFER, dest);
		 glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, size);
		 glBindBuffer(GL_COPY_READ_BUFFER, 0);
		 glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
	 }

	 void DeleteBuffer(GLuint buffer) {
		 glDeleteBuffers(1, &buffer);
	 }

	 void IncreaseMemorySize(int new_size) {
		 GLuint new_memory = CreateBuffer(new_size);
		 CopyBuffer(memory, new_memory, size);
		 DeleteBuffer(memory);
		 memory = new_memory;
		 size = new_size;
	 }

	 void SetDataAtOffset(uint offset, const vector<uint>& data) {
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory);
		 glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, data.size() * sizeof(uint), data.data());
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	 }

	 TensorMemory* Allocate(const vector<int>& shape, const DataType type = DataType::Float) override {
		 int size = GetLinearSize(shape);

		 if (size == 0) {
			 throw invalid_argument("Trying to allocate a tensor with size 0");
		 }

		 Frame* frame = allocator.AllocateFrame(size);
		 // reserve space in memory if needed
		 if ((int)frame->end > size) {
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

	 vector<uint> Readback(const TensorMemory* mem) override {
		 vector<uint> data;
		 data.resize(mem->GetSize());
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory);
		 glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, mem->frame->start, data.size() * sizeof(uint), data.data());
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 return data;
	 }

	 uint ReadbackValue(const TensorMemory* mem, uint index) override {
		 uint data;
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory);
		 glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, mem->frame->start + index, sizeof(uint), &data);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		 return data;
	 }

	 void Writeback(const TensorMemory* mem, const vector<uint>& data) override {
		 SetDataAtOffset(mem->frame->start, data);
	 }

	 void WritebackValue(const TensorMemory* mem, uint index, uint value) override {
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory);
		 glBufferSubData(GL_SHADER_STORAGE_BUFFER, mem->frame->start + index, sizeof(uint), &value);
		 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	 }

	 ~OpenGLMemoryManager() {
		 DeleteBuffer(memory);
	 }
};


}  // namespace TensorFrost