#pragma once

#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../KernelManager.h"

namespace TensorFrost {

class OpenGLKernelManager : public KernelManager {
	unordered_map<size_t, GLuint> kernel_map;
	const int WORK_GROUP_SIZE = 256;
	GLuint ubo;
 public:
	OpenGLKernelManager() {
        glGenBuffers(1, &ubo);
		//allocate sizeof(uint32_t) * 32 bytes
		glBindBuffer(GL_UNIFORM_BUFFER, ubo);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(uint32_t) * 32, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }

	void UpdateUBO(const uint32_t* data, size_t size) {
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(uint32_t) * size, data);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }
	
	GLuint createComputeShader(const std::string& source) {
		GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
		const char* src = source.c_str();
		glShaderSource(shader, 1, &src, nullptr);
		glCompileShader(shader);

		// Check for compilation errors
		GLint success;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success) {
			GLchar infoLog[512];
			glGetShaderInfoLog(shader, 512, nullptr, infoLog);
			throw std::runtime_error("TensorFrost: Error compiling shader: " + source + "\n" + std::string(infoLog));
		}

		return shader;
	}

	GLuint createShaderProgram(const std::string& computeShaderSource) {
		GLuint computeShader = createComputeShader(computeShaderSource);
		GLuint program = glCreateProgram();
		glAttachShader(program, computeShader);
		glLinkProgram(program);

		// Check for linking errors
		GLint success;
		glGetProgramiv(program, GL_LINK_STATUS, &success);
		if (!success) {
			GLchar infoLog[512];
			glGetProgramInfoLog(program, 512, nullptr, infoLog);
			throw std::runtime_error("TensorFrost: Error linking program: " + std::string(infoLog));
		}

		glDeleteShader(computeShader);
		return program;
	}

	void CompileKernel(Kernel* kernel) 
	{
	#ifndef NDEBUG
		cout << "Compiling kernel \n" << kernel->full_generated_code_ << endl;
	#endif
		//print out source if debug is enabled
		GLuint program = createShaderProgram(kernel->full_generated_code_);
		kernel_map[kernel->kernel_id_] = program;
	}

	//Get uniform location
	GLint getUniformLocation(GLuint program, const std::string& name) {
		GLint location = glGetUniformLocation(program, name.c_str());
		if (location == -1) {
			throw std::runtime_error("OpenGL error: uniform " + name + " not found");
		}
		return location;
	}

	//Get attribute location
	GLint getAttribLocation(GLuint program, const std::string& name) {
		GLint location = glGetAttribLocation(program, name.c_str());
		if (location == -1) {
			throw std::runtime_error("OpenGL error: attribute " + name + " not found");
		}
		return location;
	}

	void DispatchKernel(TFDispatchInfo info) override
	{
		GLuint program = kernel_map[info.kernel_id];
		Kernel* kernel = GetKernel(info.kernel_id);
		glUseProgram(program);

		#ifndef NDEBUG
		// validate the program
		glValidateProgram(program);
		GLint success;
		glGetProgramiv(program, GL_VALIDATE_STATUS, &success);
		if (!success) {
			GLchar infoLog[512];
			glGetProgramInfoLog(program, 512, nullptr, infoLog);
			throw std::runtime_error("OpenGL error: program validation failed: " + std::string(infoLog));
		}
		#endif

		// Set uniforms
		if (info.read_write_count == 0) throw std::runtime_error("No tensors provided to kernel");

		//bind all memory buffers
		for (size_t i = 0; i < info.read_write_count; i++) {
			GLuint buffer = ((TFOpenGLBuffer*)info.read_write_tensors[i].buffer)->GetNative();
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (GLuint)i, buffer);
		}

		if (info.variable_count > 0)
		{
			UpdateUBO(info.variables, info.variable_count);
		}

		// Bind the UBO
		glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);

		// Dispatch the kernel
		glDispatchCompute((GLuint)info.work_group_count, 1, 1);

		// Wait for the kernel to finish
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

		// Unbind the memory buffer
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);

		// Unbind the program
		glUseProgram(0);

		// Check for errors
		GLenum error = glGetError();
		if (error != GL_NO_ERROR) {
			throw std::runtime_error("OpenGL error: " + std::to_string(error));
		}
	}

	void FreeKernel(int kernel_id)
	{
		GLuint program = kernel_map[kernel_id];
		glDeleteProgram(program);
		kernel_map.erase(kernel_id);
	}

	void FreeAllKernels()
	{
		for (auto& kernel : kernel_map) {
			glDeleteProgram(kernel.second);
		}
		kernel_map.clear();
	}

	~OpenGLKernelManager()
	{
		FreeAllKernels();
		glDeleteBuffers(1, &ubo);
	}
};

}  // namespace TensorFrost