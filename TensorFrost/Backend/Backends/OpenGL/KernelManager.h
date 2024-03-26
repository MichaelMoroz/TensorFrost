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
	unordered_map<int, GLuint> kernel_map;
	const int WORK_GROUP_SIZE = 256;
 public:
	
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
			std::cerr << "TensorFrost: Error compiling shader: " << infoLog << std::endl;
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
			std::cerr << "TensorFrost: Error linking program: " << infoLog << std::endl;
		}

		glDeleteShader(computeShader);
		return program;
	}

	void CompileKernel(Kernel* kernel) 
	{
		if (kernel->indexing_mode_ != KernelIndexingMode::Linear) {
			std::cerr << "TensorFrost: OpenGL backend only supports linear indexing mode" << std::endl;
			return;
		}
		GLuint program = createShaderProgram(kernel->generated_code_);
		kernel_map[kernel->kernel_id_] = program;
	}

	void DispatchKernel(DispatchInfo info) override
	{
		GLuint program = kernel_map[info.kernel_id];
		glUseProgram(program);

		//compute number of threads
		int thread_count = 1;
		for (int i = 0; i < (int)info.dispatch_dim; i++) {
			thread_count *= info.dispatch_shape[i];
		}

		//compute number of work groups
		int work_group_count = (thread_count + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;

		// Get memory
		OpenGLMemoryManager* memory_manager =
		    (OpenGLMemoryManager*)global_memory_manager;
		GLuint memory_ssbo = memory_manager->memory;

		// Bind the memory buffer to the shader
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, memory_ssbo);

		// Set uniforms
		// Set the number of threads
		glUniform1i(glGetUniformLocation(program, "dispatch_size"), thread_count);
		// Set offsets uniform array
		std::vector<int> offsets;
		for (int i = 0; i < (int)info.tensor_count; i++) {
			offsets.push_back(info.tensors[i].offset);
		}
		glUniform1iv(glGetUniformLocation(program, "off"), info.tensor_count, offsets.data());
		// Set variables uniform array
		glUniform1iv(glGetUniformLocation(program, "var"), info.variable_count, (int*)info.variables);

		// Dispatch the kernel
		glDispatchCompute(work_group_count, 1, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

		// Unbind the memory buffer
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);

		// Unbind the program
		glUseProgram(0);

		// Check for errors
		GLenum error = glGetError();
		if (error != GL_NO_ERROR) {
			std::cerr << "OpenGL error: " << error << std::endl;
		}

		// Wait for the kernel to finish
		glFinish();
	}
};

}  // namespace TensorFrost