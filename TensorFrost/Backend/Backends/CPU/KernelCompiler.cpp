#include "KernelCompiler.h"
#include <sstream>

namespace TensorFrost {

std::string C_COMPILER_PATH = "";

bool RunCompiler(TCHAR* tempPath, TCHAR* dllName)
{
	std::string compilerPath = C_COMPILER_PATH;
	cout << "CompilerPath: " << compilerPath << endl;
    //char command[512];
	//sprintf(command, "%s -shared temp/generated_lib.c -o temp/generated_lib.dll", \
         compilerPath.c_str());

	std::basic_stringstream<TCHAR> ss;
	//ss << compilerPath << " -g -shared " << tempPath << "generated_lib.c -o " << dllName;
	//ss << compilerPath << " /LD /Zi " << tempPath << "generated_lib.c /Fe:" << dllName; // MSVC
	ss << compilerPath << " /LD " << tempPath << "generated_lib.c /Fe:" << dllName;  // MSVC
	std::basic_string<TCHAR> command = ss.str();

	cout << "Command: " << command << endl;

	STARTUPINFO si;
	PROCESS_INFORMATION pi;
	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	ZeroMemory(&pi, sizeof(pi));

	// Start the child process
	if (!CreateProcess(NULL,     // No module name (use command line)
	                   command.data(),  // Command line
	                   NULL,     // Process handle not inheritable
	                   NULL,     // Thread handle not inheritable
	                   FALSE,    // Set handle inheritance to FALSE
	                   0,        // No creation flags
	                   NULL,     // Use parent's environment block
	                   NULL,     // Use parent's starting directory
	                   &si,      // Pointer to STARTUPINFO structure
	                   &pi       // Pointer to PROCESS_INFORMATION structure
	                   )) {
		throw std::runtime_error("Compiler error: cannot create compiler process");
	}

	// Wait until child process exits
	WaitForSingleObject(pi.hProcess, INFINITE);
	
	// Check for compiler errors
	DWORD exitCode;
	GetExitCodeProcess(pi.hProcess, &exitCode);
	if (exitCode != 0) {
		throw std::runtime_error(
		    "Compiler error: compiler exited with non-zero exit code (Error "
		    "code: " +
		    to_string(exitCode) + ")");
	}

	// Close process and thread handles
	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);

	return true;
}

void CompileKernelLibrary(string sourceCode, TCHAR* tempPath, TCHAR* dllName) 
{
	// Append a file name to the tempPath
	std::basic_stringstream<TCHAR> ss;
	ss << tempPath << "generated_lib.c";  // Choose an appropriate file name
	std::basic_string<TCHAR> fullFilePath = ss.str();

	std::string filePath(fullFilePath);

	cout << "Source path: " << filePath << endl;

	// Write the generated source code to a file
	std::ofstream outFile(filePath);
	if (!outFile) {
		throw std::runtime_error(
		    "Compiler error: cannot open file for writing generated source code");
	}
	outFile << sourceCode;
	outFile.close();

	RunCompiler(tempPath, dllName);
}

typedef unsigned int uint;
typedef void (*kernel_func)(uint*, uint*, uint*, uint);

void CompileAndLoadKernel(Program* program) 
{
	TCHAR tempPath[MAX_PATH];
	DWORD pathLength = GetTempPath(MAX_PATH, tempPath);

	if (pathLength == 0) {
		throw std::runtime_error("Compiler error: cannot get temp path");
	}

	// Create a temporary library name
	TCHAR tempFileName[MAX_PATH];
	if (!GetTempFileName(tempPath, TEXT("lib"), 0, tempFileName)) {
		throw std::runtime_error("Compiler error: cannot create temp file");
	}

	cout << "Temp file: " << tempFileName << endl;

	// Generate C code
	pair<string, vector<string>> source_names = GenerateC(*program->ir_);
	string sourceCode = source_names.first;
	vector<string> kernel_names = source_names.second;

	// Compile the library
    CompileKernelLibrary(sourceCode, tempPath, tempFileName);

	// Load the library
	HMODULE lib_handle = LoadLibrary(tempFileName);
	if (!lib_handle) {
		throw std::runtime_error("Compiler error: cannot load generated library");
	}

	// Create lambda function to free the library
	program->unload_callback = [lib_handle]() {
		if (!FreeLibrary(lib_handle)) {
			std::cerr << "Cannot free library: " << GetLastError() << '\n';
		}
	};

	// Load symbols for each kernel
	int i = 0;
	for (int k = 0; k < program->kernels_.size(); k++) {
		Kernel* kernel = &program->kernels_[k];
		if (kernel->type_ != KernelType::Compute) {
			continue;
		}

		string kernel_name = kernel_names[i];
		string symbol_name = kernel_name;

		kernel_func kernel_callback = (kernel_func)GetProcAddress(lib_handle, symbol_name.c_str());

		if (!kernel_callback) {
			throw std::runtime_error("Compiler error: cannot load kernel function");
		}

		kernel->execute_callback = [kernel_callback](
		                              TensorMemoryManager* memory_manager,
		                                   vector<uint> variables,
		                                   vector<uint> offsets, uint threads) {
			// get CPU memory manager
			CPU_MemoryManager* cpu_memory_manager = dynamic_cast<CPU_MemoryManager*>(memory_manager);
			if (!cpu_memory_manager) {
				throw std::runtime_error("Cannot execute kernel on non-CPU memory manager");
			}
			// get memory
			uint* memory = cpu_memory_manager->memory.data();
			// execute kernel
			kernel_callback(variables.data(), offsets.data(), memory, threads);
		};

		i++;
	}

	cout << "Successfully compiled and loaded kernel library." << endl;
}

}  // namespace TensorFrost

