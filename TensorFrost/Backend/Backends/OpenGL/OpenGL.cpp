#include "OpenGL.h"

namespace TensorFrost {

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
}

void StopOpenGL() {
	glfwTerminate();

	global_window = nullptr;
}

}  // namespace TensorFrost