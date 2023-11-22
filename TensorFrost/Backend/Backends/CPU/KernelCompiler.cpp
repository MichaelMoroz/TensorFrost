#include "KernelCompiler.h"

namespace TensorFrost {

std::string C_COMPILER_PATH = "";

bool compileLibraryWithCreateProcess() {
	std::string compilerPath = C_COMPILER_PATH;
	cout << "compilerPath: " << compilerPath << endl;
    char command[512];
    sprintf(command, "%s -shared temp/generated_lib.c -o temp/generated_lib.dll", compilerPath.c_str());

	cout << "command: " << command << endl;

	STARTUPINFO si;
	PROCESS_INFORMATION pi;
	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	ZeroMemory(&pi, sizeof(pi));

	// Start the child process
	if (!CreateProcess(NULL,     // No module name (use command line)
	                   command,  // Command line
	                   NULL,     // Process handle not inheritable
	                   NULL,     // Thread handle not inheritable
	                   FALSE,    // Set handle inheritance to FALSE
	                   0,        // No creation flags
	                   NULL,     // Use parent's environment block
	                   NULL,     // Use parent's starting directory
	                   &si,      // Pointer to STARTUPINFO structure
	                   &pi       // Pointer to PROCESS_INFORMATION structure
	                   )) {
		std::cerr << "CreateProcess failed (" << GetLastError() << ")\n";
		return false;
	}

	// Wait until child process exits
	WaitForSingleObject(pi.hProcess, INFINITE);

	// Close process and thread handles
	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);

	return true;
}

void compileLibrary() {
	// Generate the C source code for the library
	const char* sourceCode = R"(
	__declspec(dllexport) int add(int a, int b) {
		return a + b;
	}
	)";

	// Write the generated source code to a file
	std::ofstream outFile("temp/generated_lib.c");
	if (!outFile) {
		std::cerr << "Error creating file generated_lib.c\n";
		return;
	}
	outFile << sourceCode;
	outFile.close();

	// Compile the generated C code into a dynamic library (DLL)
	// Note: Adjust the command as needed for your compiler and environment
	compileLibraryWithCreateProcess();
}

typedef int (*add_func)(int, int);

void loadLibraryWin() {
	compileLibrary();  // Compile the library

	// Load the library
	HMODULE lib_handle = LoadLibrary(TEXT("temp/generated_lib.dll"));
	if (!lib_handle) {
		std::cerr << "Cannot load library: " << GetLastError() << '\n';
		return;
	}

	// Load the symbol
	add_func add = (add_func)GetProcAddress(lib_handle, "add");
	if (!add) {
		std::cerr << "Cannot load symbol 'add': " << GetLastError() << '\n';
		FreeLibrary(lib_handle);
		return;
	}

	// Use the function
	std::cout << "add(2, 3) = " << add(2, 3) << '\n';

	// Close the library
	FreeLibrary(lib_handle);
}

}  // namespace TensorFrost

