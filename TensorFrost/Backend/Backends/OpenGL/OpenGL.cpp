#include "OpenGL.h"

namespace TensorFrost {

GLFWwindow* global_window = nullptr;

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
}

void StopOpenGL() {
	glfwTerminate();

	global_window = nullptr;
}

}  // namespace TensorFrost