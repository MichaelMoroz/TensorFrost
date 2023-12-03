#include "KernelCompiler.h"

#include <sstream>

namespace TensorFrost {

std::string c_compiler_path;

bool RunCompiler(TCHAR* tempPath, TCHAR* dllName) {
	std::string compiler_path = c_compiler_path;
	cout << "CompilerPath: " << compiler_path << endl;
	// char command[512];
	// sprintf(command, "%s -shared temp/generated_lib.c -o
	// temp/generated_lib.dll",
	//     compilerPath.c_str());

	std::basic_stringstream<TCHAR> ss;
	// ss << compilerPath << " -g -shared " << tempPath << "generated_lib.c -o "
	// << dllName; ss << compilerPath << " /LD /Zi " << tempPath <<
	// "generated_lib.c /Fe:" << dllName; // MSVC
	ss << compiler_path << " /LD " << tempPath
	   << "generated_lib.cpp /Fe:" << dllName;  // MSVC
	std::basic_string<TCHAR> command = ss.str();

	cout << "Command: " << command << endl;

	STARTUPINFO si;
	PROCESS_INFORMATION pi;
	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	ZeroMemory(&pi, sizeof(pi));

	// Start the child process
	if (!CreateProcess(nullptr,         // No module name (use command line)
	                   command.data(),  // Command line
	                   nullptr,         // Process handle not inheritable
	                   nullptr,         // Thread handle not inheritable
	                   FALSE,           // Set handle inheritance to FALSE
	                   0,               // No creation flags
	                   nullptr,         // Use parent's environment block
	                   nullptr,         // Use parent's starting directory
	                   &si,             // Pointer to STARTUPINFO structure
	                   &pi  // Pointer to PROCESS_INFORMATION structure
	                   )) {
		throw std::runtime_error("Compiler error: cannot create compiler process");
	}

	// Wait until child process exits
	WaitForSingleObject(pi.hProcess, INFINITE);

	// Check for compiler errors
	DWORD exit_code;
	GetExitCodeProcess(pi.hProcess, &exit_code);
	if (exit_code != 0) {
		throw std::runtime_error(
		    "Compiler error: compiler exited with non-zero exit code (Error "
		    "code: " +
		    to_string(exit_code) + ")");
	}

	// Close process and thread handles
	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);

	return true;
}

void CompileKernelLibrary(const string& sourceCode, TCHAR* tempPath,
                          TCHAR* dllName) {
	// Append a file name to the tempPath
	std::basic_stringstream<TCHAR> ss;
	ss << tempPath << "generated_lib.cpp";  // Choose an appropriate file name
	std::basic_string<TCHAR> full_file_path = ss.str();

	const std::string& file_path(full_file_path);

	cout << "Source path: " << file_path << endl;

	// Write the generated source code to a file
	std::ofstream out_file(file_path);
	if (!out_file) {
		throw std::runtime_error(
		    "Compiler error: cannot open file for writing generated source code");
	}
	out_file << sourceCode;
	out_file.close();

	RunCompiler(tempPath, dllName);
}

using uint = unsigned int;
using kernel_func = void (*)(uint*, uint*, uint*, uint*);

void CompileAndLoadKernel(Program* program) {
	TCHAR temp_path[MAX_PATH];
	DWORD path_length = GetTempPath(MAX_PATH, temp_path);

	if (path_length == 0) {
		throw std::runtime_error("Compiler error: cannot get temp path");
	}

	// Create a temporary library name
	TCHAR temp_file_name[MAX_PATH];
	if (!GetTempFileName(temp_path, TEXT("lib"), 0, temp_file_name)) {
		throw std::runtime_error("Compiler error: cannot create temp file");
	}

	cout << "Temp file: " << temp_file_name << endl;

	// Generate C code
	pair<string, vector<string>> source_names = GenerateC(program);
	string source_code = source_names.first;
	vector<string> kernel_names = source_names.second;

	// Compile the library
	CompileKernelLibrary(source_code, temp_path, temp_file_name);

	// Load the library
	HMODULE lib_handle = LoadLibrary(temp_file_name);
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
	for (auto& k : program->kernels_) {
		Kernel* kernel = &k;
		if (kernel->type_ != KernelType::Compute) {
			continue;
		}

		string kernel_name = kernel_names[i];
		const string& symbol_name = kernel_name;

		auto kernel_callback = reinterpret_cast<kernel_func>(
		    GetProcAddress(lib_handle, symbol_name.c_str()));

		if (!kernel_callback) {
			throw std::runtime_error("Compiler error: cannot load kernel function");
		}

		kernel->execute_callback = [kernel_callback](
		                               TensorMemoryManager* memory_manager,
		                               vector<uint> variables, vector<uint> offsets,
		                               vector<uint> shape) {
			// get CPU memory manager
			auto* cpu_memory_manager =
			    dynamic_cast<CpuMemoryManager*>(memory_manager);
			if (!cpu_memory_manager) {
				throw std::runtime_error(
				    "Cannot execute kernel on non-CPU memory manager");
			}
			// get memory
			uint* memory = cpu_memory_manager->memory.data();
			// execute kernel
			kernel_callback(variables.data(), offsets.data(), memory, shape.data());
		};

		i++;
	}

	cout << "Successfully compiled and loaded kernel library." << endl;
}

}  // namespace TensorFrost
