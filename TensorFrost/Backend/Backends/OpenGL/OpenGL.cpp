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
bool window_open = false;
ImGuiIO* io;

void GLAPIENTRY DebugCallback(GLenum source, GLenum type, GLuint id,
                              GLenum severity, GLsizei length,
                              const GLchar* message, const void* userParam) {
	// Output or log the debug message
	std::cerr << "OpenGL Debug: source=" << source << ", type=" << type
	          << ", id=" << id << ", severity=" << severity << endl;
	std::cerr << "Message: " << message << endl << endl;
}


// Window close callback function
void WindowCloseCallback(GLFWwindow* window) {
	window_open = false;

	// Instead of closing, hide the window
	glfwHideWindow(window);

	// Prevent the window from closing
	glfwSetWindowShouldClose(window, GLFW_FALSE);
}

void WindowSizeCallback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

void ImguiNewFrame() {
	if (global_window == nullptr) {
		throw std::runtime_error("Window: OpenGL not initialized");
	}

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

void ImguiRender() {
	if (global_window == nullptr) {
		throw std::runtime_error("Window: OpenGL not initialized");
	}

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

std::string last_error;

void error_callback(int error, const char* description) {
	last_error = std::string(description);
}

//#define OPENGL_DEBUG

void StartOpenGL() {
	glfwSetErrorCallback(error_callback);

	if (!glfwInit()) {
		throw std::runtime_error("Failed to initialize GLFW: " + last_error);
	}

	// Make window invisible
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	global_window = glfwCreateWindow(800, 600, "TensorFrost", nullptr, nullptr);

	if (!global_window) {
		int code = glfwGetError(nullptr);
		glfwTerminate();
		throw std::runtime_error("Failed to create window (error " + std::to_string(code) + "): " + last_error);
	}

	glfwMakeContextCurrent(global_window);


	int version = gladLoadGL(glfwGetProcAddress);
	if (version == 0) {
		throw std::runtime_error("Failed to load OpenGL");
	}

	// Successfully loaded OpenGL
	printf("Loaded OpenGL %d.%d\n", GLAD_VERSION_MAJOR(version),
	       GLAD_VERSION_MINOR(version));

	// Print the renderer string, which usually contains the GPU's name
	const GLubyte* renderer = glGetString(GL_RENDERER);
	const GLubyte* vendor = glGetString(GL_VENDOR);
	printf("Renderer: %s\nVendor: %s\n", renderer, vendor);

	// Print the max buffer size
	GLint bufferSize;
	glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &bufferSize);
	printf("Max available buffer size, MB: %d\n", bufferSize / 1024 / 1024);

	// Print max SSBO bindings
	GLint max_ssbo_bindings;
	glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &max_ssbo_bindings);
	printf("Maximum SSBO bindings supported: %d\n", max_ssbo_bindings);

	// Enable debug output
	#if !defined(NDEBUG) || defined(OPENGL_DEBUG)
	if (GLAD_GL_KHR_debug) {
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(DebugCallback, nullptr);
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_HIGH, 0, nullptr,
		                      GL_TRUE);
	}
	#endif

	// Create the shader program
	quad_program = CreateProgram(vertex_shader, fragment_shader);

	glfwSetWindowCloseCallback(global_window, WindowCloseCallback);
	glfwSetWindowSizeCallback(global_window, WindowSizeCallback);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	io = &ImGui::GetIO(); (void)*io;
	io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	
	ImGui_ImplGlfw_InitForOpenGL(global_window, true);
	ImGui_ImplOpenGL3_Init("#version 430");

	ImguiNewFrame();
}

void StopOpenGL() {
	if (global_window == nullptr) {
		throw std::runtime_error("OpenGL not initialized");
	}

	glfwDestroyWindow(global_window);
	glfwTerminate();

	glDeleteProgram(quad_program);
	global_window = nullptr;

	// Cleanup ImGui
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void ShowWindow(int width, int height, const char* title) {
	window_open = true;

	if(global_window == nullptr) {
		throw std::runtime_error("Window: OpenGL not initialized");
	}

	glfwSetWindowSize(global_window, width, height);
	glfwSetWindowTitle(global_window, title);
	glfwShowWindow(global_window);

	//reset viewport
	glViewport(0, 0, width, height);
}

void HideWindow() {
	if(global_window == nullptr) {
		throw std::runtime_error("Window: OpenGL not initialized");
	}

	window_open = false;
	glfwHideWindow(global_window);
}

void Finish() {
	glFinish();
}

void RenderFrame(const TFTensor* tensor) {
	if (global_window == nullptr) {
		throw std::runtime_error("RenderFrame: OpenGL not initialized");
	}

	// Clear the screen
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	if(tensor != nullptr)
	{
		//check if tensor is 2d + 3 channels
		if (tensor->dim != 3 || tensor->shape[2] != 3) {
			throw std::runtime_error("Window: Render tensor must be of shape (height, width, 3)");
		}

		//check if tensor is float32 (TODO: use int8 instead)
		if (tensor->format != TFTypeFloat32) {
			throw std::runtime_error("Window: Render tensor must be of type float32");
		}

		GLuint ssbo = ((TFOpenGLBuffer*)tensor->buffer)->GetNative();
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

		glUseProgram(quad_program);

		// Set the uniforms
		int offset = 0;
		int width = (int)tensor->shape[1];
		int height = (int)tensor->shape[0];
		glUniform1i(glGetUniformLocation(quad_program, "offset"), offset);
		glUniform1i(glGetUniformLocation(quad_program, "width"), width);
		glUniform1i(glGetUniformLocation(quad_program, "height"), height);

		// Draw the quad
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
		glUseProgram(0);
	}

	ImguiRender();

	// Swap the buffers
	glfwSwapBuffers(global_window);
	glfwPollEvents();

	ImguiNewFrame();
}

bool WindowShouldClose() { return !window_open; }

pair<double, double> GetMousePosition() {
	double x, y;
	glfwGetCursorPos(global_window, &x, &y);
	return {x, y};
}

pair<int, int> GetWindowSize() {
	int width, height;
	glfwGetWindowSize(global_window, &width, &height);
	return {width, height};
}

bool IsMouseButtonPressed(int button) {
	//if pressed in imgui, return false
	if (io->WantCaptureMouse) {
		return false;
	}
	return glfwGetMouseButton(global_window, button) == GLFW_PRESS;
}

bool IsKeyPressed(int key) {
	return glfwGetKey(global_window, key) == GLFW_PRESS;
}

void ImGuiBegin(std::string name) {
	ImGui::Begin(name.c_str());
}

void ImGuiEnd() {
	ImGui::End();
}

void ImGuiText(const std::string& text) {
	ImGui::Text("%s", text.c_str());
}

void ImGuiSlider(std::string text, int* value, int min, int max) {
	ImGui::SliderInt(text.c_str(), value, min, max);
}

void ImGuiSlider(std::string text, float* value, float min, float max) {
	ImGui::SliderFloat(text.c_str(), value, min, max, "%.5f");
}

bool ImGuiButton(std::string text) {
	return ImGui::Button(text.c_str());
}

void ImGuiPlotLines(const char *label, const float *values, int values_count, int values_offset,
	const char *overlay_text, float scale_min, float scale_max, ImVec2 graph_size, int stride) {
	ImGui::PlotLines(label, values, values_count, values_offset, overlay_text, scale_min, scale_max, graph_size, stride);
}

bool ImGuiCheckbox(std::string text, bool* value) {
	return ImGui::Checkbox(text.c_str(), value);
}

void ImGuiScaleAllSizes(float scale) {
	ImGuiStyle& style = ImGui::GetStyle();
	style.ScaleAllSizes(scale);
}

void ImGuiAddBackgroundText(const std::string &text, const ImVec2 &pos, const ImVec4 &color) {
	ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
	draw_list->AddText(pos, ImGui::GetColorU32(color), text.c_str());
}

void ImGuiColorPicker3(const std::string &text, float *color) {
	ImGui::ColorPicker3(text.c_str(), color);
}

void ImGuiColorPicker4(const std::string &text, float *color) {
	ImGui::ColorPicker4(text.c_str(), color);
}

void StartDebugRegion(const std::string& name) {
	glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, (GLsizei)name.size(), name.c_str());
}

void EndDebugRegion() { glPopDebugGroup(); }

}  // namespace TensorFrost