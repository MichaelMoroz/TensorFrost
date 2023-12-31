#include "KernelCompiler.h"

#include <sstream>

namespace TensorFrost {

std::string kernel_compile_options;

bool RunCompiler(char* tempPath, char* dllName) {
	cout << "Compile options: " << kernel_compile_options << endl;
	std::basic_stringstream<char> ss;

#if defined(_WIN32)
	//what the fu..
	ss << "powershell -command \"$VisualStudioPath = & \\\"${Env:ProgramFiles(x86)}\\Microsoft Visual Studio\\Installer\\vswhere.exe\\\" -latest -products * -property installationPath; & cmd.exe /C \\\"\"\\\"\\\"$VisualStudioPath\\VC\\Auxiliary\\Build\\vcvarsall.bat\\\"\\\" x64 && cl " 
	   << kernel_compile_options << " /LD " << tempPath
	   << "generated_lib.cpp /Fe:" << dllName 
	   << "\"\"\\\"\"";  // MSVC
#else
	ss << "g++ " << kernel_compile_options << " -shared -fPIC " << tempPath
	   << "generated_lib.cpp -o " << dllName;  // GCC
#endif

	std::basic_string<char> command = ss.str();

	cout << "Command: " << command << endl;

	// Run the compiler
#if defined(_WIN32)
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
		throw std::runtime_error(std::string("Compiler error: cannot create compiler process. Command line: ") + command.data() + "\n");
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
#else
	//Use linux execl
	pid_t pid = fork();
	if (pid == 0) {
		execl("/bin/sh", "sh", "-c", command.data(), nullptr);
		exit(127);
	} else {
		int status;
		waitpid(pid, &status, 0);
		if (status != 0) {
			throw std::runtime_error(
			    "Compiler error: compiler exited with non-zero exit code (Error "
			    "code: " +
			    to_string(status) + ")");
		}
	}
#endif

	return true;
}

void CompileKernelLibrary(const string& sourceCode, char* tempPath,
                          char* dllName) {
	// Append a file name to the tempPath
	std::basic_stringstream<char> ss;
	ss << tempPath << "generated_lib.cpp";  // Choose an appropriate file name
	std::basic_string<char> full_file_path = ss.str();

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
#if defined(_WIN32)
	char temp_path[MAX_PATH];
	DWORD path_length = GetTempPath(MAX_PATH, temp_path);

	if (path_length == 0) {
		throw std::runtime_error("Compiler error: cannot get temp path");
	}

	// Create a temporary library name
	char temp_file_name[MAX_PATH];
	if (!GetTempFileName(temp_path, TEXT("lib"), 0, temp_file_name)) {
		throw std::runtime_error("Compiler error: cannot create temp file");
	}
#else
	char temp_path[] = "/tmp/";
	char filename_template[] = "/tmp/mytempXXXXXX";
	char* temp_file_name = mktemp(filename_template);
	if (!temp_file_name) {
		throw std::runtime_error("Compiler error: cannot create temp file");
	}
#endif

	cout << "Temp file: " << temp_file_name << endl;

	// Generate C code
	pair<string, vector<string>> source_names = GenerateC(program);
	string source_code = source_names.first;
	vector<string> kernel_names = source_names.second;

	// Compile the library
	CompileKernelLibrary(source_code, temp_path, temp_file_name);

	// Load the library
	#if defined(_WIN32)
	HMODULE lib_handle = LoadLibrary(temp_file_name);
	if (!lib_handle) {
		throw std::runtime_error("Compiler error: cannot load generated library");
	}
	#else
	void* lib_handle = dlopen(temp_file_name, RTLD_LAZY);
	if (!lib_handle) {
		throw std::runtime_error("Compiler error: cannot load generated library");
	}
	#endif

	// Create lambda function to free the library
	program->unload_callback = [lib_handle]() {
		#if defined(_WIN32)
		if (!FreeLibrary(lib_handle)) {
			std::cerr << "Cannot free library: " << GetLastError() << '\n';
		}
		#else
		if (dlclose(lib_handle)) {
			std::cerr << "Cannot free library: " << dlerror() << '\n';
		}
		#endif
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

		// Load the symbol
		#if defined(_WIN32)
		auto kernel_callback = reinterpret_cast<kernel_func>(
		    GetProcAddress(lib_handle, symbol_name.c_str()));
		#else
		auto kernel_callback = reinterpret_cast<kernel_func>(
		    dlsym(lib_handle, symbol_name.c_str()));
		#endif

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
