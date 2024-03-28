#include "OpenGL.h"

namespace TensorFrost {

string vertex_shader = R"(
#version 430 core
out vec2 texCoords;

void main() {
    const vec2 pos[6] = vec2[](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
                               vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
    gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
    texCoords = pos[gl_VertexID] * 0.5 + 0.5;
}
)";

string fragment_shader = R"(
#version 430 core
in vec2 texCoords;
out vec4 FragColor;

layout(std430, binding = 0) buffer memory {
  uint mem[];
};

uniform int offset;
uniform int width;
uniform int height;

void main() {
	ivec2 pixel = ivec2(texCoords.x * width, texCoords.y * height);
	int pixel_offset = pixel.y * width + pixel.x;
	int cur_offset = pixel_offset * 3 + offset;
    float r = uintBitsToFloat(mem[cur_offset]);
	float g = uintBitsToFloat(mem[cur_offset + 1]);
	float b = uintBitsToFloat(mem[cur_offset + 2]);
	FragColor = vec4(r, g, b, 1.0);
}
)";

GLuint CreateShader(const string& source, GLenum type) {
	GLuint shader = glCreateShader(type);
	const char* src = source.c_str();
	glShaderSource(shader, 1, &src, nullptr);
	glCompileShader(shader);

	int success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetShaderInfoLog(shader, 512, nullptr, infoLog);
		throw std::runtime_error("Failed to compile shader: " + string(infoLog));
	}

	return shader;
}

GLuint CreateProgram(const string& vertexSource, const string& fragmentSource) {
	GLuint vertexShader = CreateShader(vertexSource, GL_VERTEX_SHADER);
	GLuint fragmentShader = CreateShader(fragmentSource, GL_FRAGMENT_SHADER);

	GLuint program = glCreateProgram();
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	glLinkProgram(program);

	int success;
	glGetProgramiv(program, GL_LINK_STATUS, &success);

	return program;
}

GLuint quad_program = 0;

GLFWwindow* global_window = nullptr;

void GLAPIENTRY DebugCallback(GLenum source, GLenum type, GLuint id,
                              GLenum severity, GLsizei length,
                              const GLchar* message, const void* userParam) {
	// Output or log the debug message
	std::cerr << "OpenGL Debug: source=" << source << ", type=" << type
	          << ", id=" << id << ", severity=" << severity << endl;
	std::cerr << "Message: " << message << endl << endl;
}

void StartOpenGL() {
	if (!glfwInit()) {
		throw std::runtime_error("Failed to initialize GLFW");
	}

	//make window invisible
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	global_window = glfwCreateWindow(800, 600, "TensorFrost", nullptr, nullptr);

	if (!global_window) {
		glfwTerminate();
		throw std::runtime_error("Failed to create window");
	}

	glfwMakeContextCurrent(global_window);

	int version = gladLoadGL(glfwGetProcAddress);
	if (version == 0) {
		throw std::runtime_error("Failed to load OpenGL");
	}

	// Successfully loaded OpenGL
	printf("Loaded OpenGL %d.%d\n", GLAD_VERSION_MAJOR(version),
	       GLAD_VERSION_MINOR(version));

	// Enable debug output
	#ifndef NDEBUG
	if (GLAD_GL_KHR_debug) {
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(DebugCallback, nullptr);
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr,
		                      GL_TRUE);
	}
	#endif

	// Create the shader program
	quad_program = CreateProgram(vertex_shader, fragment_shader);
}

void StopOpenGL() {
	glfwTerminate();

	global_window = nullptr;
}

void ShowWindow(int width, int height, const char* title) {
	glfwSetWindowSize(global_window, width, height);
	glfwSetWindowTitle(global_window, title);
	glfwShowWindow(global_window);
}

void HideWindow() {
	glfwHideWindow(global_window);
}

void RenderFrame(const TensorMemory& tensor) {
	//check if tensor is 2d + 3 channels
	if (tensor.shape.size() != 3 && tensor.shape[2] != 3) {
		throw std::runtime_error("Tensor must be 2D with 3 channels to render");
	}

	// Clear the screen
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	GLuint ssbo = ((OpenGLMemoryManager*)global_memory_manager)->memory;
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

	glUseProgram(quad_program);

	// Set the uniforms
	int offset = tensor.frame->start;
	int width = tensor.shape[1];
	int height = tensor.shape[0];
	glUniform1i(glGetUniformLocation(quad_program, "offset"), offset);
	glUniform1i(glGetUniformLocation(quad_program, "width"), width);
	glUniform1i(glGetUniformLocation(quad_program, "height"), height);

	// Draw the quad
	glDrawArrays(GL_TRIANGLES, 0, 6);

	// Swap the buffers
	glfwSwapBuffers(global_window);
	glfwPollEvents();

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
	glUseProgram(0);
}

bool WindowShouldClose() { return glfwWindowShouldClose(global_window); }

}  // namespace TensorFrost