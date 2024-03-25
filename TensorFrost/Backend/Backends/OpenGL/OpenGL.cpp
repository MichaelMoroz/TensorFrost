#include "OpenGL.h"

namespace TensorFrost {

GLFWwindow* global_window = nullptr;

void StartOpenGL() {
	if (!glfwInit()) {
		throw std::runtime_error("Failed to initialize GLFW");
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	global_window = glfwCreateWindow(800, 600, "TensorFrost", nullptr, nullptr);

	if (!global_window) {
		glfwTerminate();
		throw std::runtime_error("Failed to create window");
	}

	glfwMakeContextCurrent(global_window);
}

void StopOpenGL() {
	glfwTerminate();

	global_window = nullptr;
}

}  // namespace TensorFrost