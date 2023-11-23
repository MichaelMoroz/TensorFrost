#include "KernelCompiler.h"

namespace TensorFrost {

std::string C_COMPILER_PATH = "";

bool RunCompiler() 
{
	std::string compilerPath = C_COMPILER_PATH;
	//cout << "compilerPath: " << compilerPath << endl;
    char command[512];
    sprintf(command, "%s -shared temp/generated_lib.c -o temp/generated_lib.dll", compilerPath.c_str());

	//cout << "command: " << command << endl;

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

void CompileKernelLibrary(string sourceCode)
{
	// Write the generated source code to a file
	std::ofstream outFile("temp/generated_lib.c");
	if (!outFile) {
		std::cerr << "Error creating file generated_lib.c\n";
		return;
	}
	outFile << sourceCode;
	outFile.close();

	RunCompiler();
}

typedef unsigned int uint;
typedef void (*kernel_func)(uint*, uint*, uint*, uint);

void CompileAndLoadKernel(Program* program) 
{
	// Generate C code
	pair<string, vector<string>> source_names = GenerateC(*program->ir_);
	string sourceCode = source_names.first;
	vector<string> kernel_names = source_names.second;

	// Compile the library
    CompileKernelLibrary(sourceCode);  

	// Load the library
	HMODULE lib_handle = LoadLibrary(TEXT("temp/generated_lib.dll"));
	if (!lib_handle) {
		std::cerr << "Cannot load library: " << GetLastError() << '\n';
		return;
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
			std::cerr << "Cannot load symbol '" << symbol_name << "': " << GetLastError() << '\n';
			FreeLibrary(lib_handle);
			return;
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

