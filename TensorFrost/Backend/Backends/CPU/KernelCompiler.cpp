#include "KernelCompiler.h"

#include <sstream>

namespace TensorFrost {

std::string kernelCompileOptions;

bool RunCompiler(char* tempPath, char* dllName, const char* sourcePath) {
    std::basic_stringstream<char> ss;
    std::string output;

#if defined(_WIN32)
    if (kernelCompileOptions.empty()) {
#ifdef NDEBUG
        kernelCompileOptions = "/O2 /fp:fast /openmp:experimental";
#else
        kernelCompileOptions = "/Zi";
#endif
    }
    ss << "powershell -command \"$VisualStudioPath = & \\\"${Env:ProgramFiles(x86)}\\Microsoft Visual Studio\\Installer\\vswhere.exe\\\" -latest -products * -property installationPath; & cmd.exe /C \\\"\"\\\"\\\"$VisualStudioPath\\VC\\Auxiliary\\Build\\vcvarsall.bat\\\"\\\" x64 && cl "
       << kernelCompileOptions << " /LD " << tempPath
       << sourcePath << " /Fe:" << dllName
       << "\"\"\\\"\"";
#else
    if (kernelCompileOptions.empty()) {
#ifdef NDEBUG
        kernelCompileOptions = "-O3 -ffast-math -fopenmp";
#else
        kernelCompileOptions = "-g";
#endif
    }
    ss << "g++ " << kernelCompileOptions << " -shared -fPIC " << tempPath
       << sourcePath << " -o " << dllName;
#endif

    std::basic_string<char> command = ss.str();

    // Run the compiler
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    HANDLE hReadPipe, hWritePipe;
    if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
        throw std::runtime_error("Failed to create pipe");
    }

    STARTUPINFO si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.hStdError = hWritePipe;
    si.hStdOutput = hWritePipe;
    si.dwFlags |= STARTF_USESTDHANDLES;

    if (!CreateProcess(nullptr, command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        throw std::runtime_error(std::string("Steps error: cannot create compiler process. Command line: ") + command.data() + "\n");
    }

    CloseHandle(hWritePipe);

    char buffer[4096];
    DWORD bytesRead;
    while (ReadFile(hReadPipe, buffer, sizeof(buffer), &bytesRead, NULL) && bytesRead != 0) {
        output.append(buffer, bytesRead);
    }

    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD exit_code;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    if (exit_code != 0) {
        throw std::runtime_error(
            "Steps error: compiler exited with non-zero exit code (Error "
            "code: " + std::to_string(exit_code) + ")\nCompiler output:\n" + output);
    }

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(hReadPipe);
#else
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }

    int status = pclose(pipe);
    if (status != 0) {
        throw std::runtime_error(
            "Steps error: compiler exited with non-zero exit code (Error "
            "code: " + std::to_string(status) + ")\nCompiler output:\n" + output);
    }
#endif

    return true;
}

void CompileKernelLibrary(const string& sourceCode, char* tempPath,
                          char* dllName, size_t program_id) {
	// Append a file name to the tempPath
	std::string source_name = "generated_lib_" + std::to_string(program_id) + ".cpp";
	std::basic_stringstream<char> ss;
	ss << tempPath << source_name;
	std::basic_string<char> full_file_path = ss.str();

	const std::string& file_path(full_file_path);

	cout << "Source path: " << file_path << endl;

	// Write the generated source code to a file
	std::ofstream out_file(file_path);
	if (!out_file) {
		throw std::runtime_error(
		    "Steps error: cannot open file for writing generated source code");
	}
	out_file << sourceCode;
	out_file.close();

	RunCompiler(tempPath, dllName, source_name.c_str());
}

void CompileAndLoadKernelModule(Program* program, size_t program_id) {
#if defined(_WIN32)
	char temp_path[MAX_PATH];
	DWORD path_length = GetTempPath(MAX_PATH, temp_path);

	if (path_length == 0) {
		throw std::runtime_error("Steps error: cannot get temp path");
	}

	// Create a temporary library name
	char temp_file_name[MAX_PATH];
	if (!GetTempFileName(temp_path, TEXT("lib"), 0, temp_file_name)) {
		throw std::runtime_error("Steps error: cannot create temp file");
	}
#else
	char temp_path[] = "/tmp/";
	char filename_template[] = "/tmp/tensorfrost_XXXXXX";
	char* temp_file_name = mktemp(filename_template);
	if (!temp_file_name) {
		throw std::runtime_error("Steps error: cannot create temp file");
	}
#endif

	cout << "Temp file: " << temp_file_name << endl;

	// Compile the library
	CompileKernelLibrary(program->generated_code_, temp_path, temp_file_name, program_id);

	// Load the library
	#if defined(_WIN32)
	HMODULE lib_handle = LoadLibrary(temp_file_name);
	if (!lib_handle) {
		throw std::runtime_error("Steps error: cannot load generated library");
	}
	#else
	void* lib_handle = dlopen(temp_file_name, RTLD_LAZY);
	if (!lib_handle) {
		throw std::runtime_error("Steps error: cannot load generated library");
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

	// Load the main function
	#if defined(_WIN32)
	auto main_callback = reinterpret_cast<main_func*>(
	    GetProcAddress(lib_handle, "main"));
	#else
	auto main_callback = reinterpret_cast<main_func*>(
	    dlsym(lib_handle, "main"));
	#endif

	if (!main_callback) {
		throw std::runtime_error("Steps error: cannot load main function");
	}

	// Set the execute callback
	program->execute_callback = *main_callback;

	// load cpu kernel functions
	if (current_backend == BackendType::CPU)
	{
		for (auto& kernel : program->kernels_) {
			#if defined(_WIN32)
			auto kernel_callback = reinterpret_cast<cpu_dispatch_func*>(
				GetProcAddress(lib_handle, kernel.kernel_name_.c_str()));
			#else
			auto kernel_callback = reinterpret_cast<cpu_dispatch_func*>(
				dlsym(lib_handle, kernel.kernel_name_.c_str()));
			#endif

			if (!kernel_callback) {
				throw std::runtime_error("Steps error: cannot load kernel function");
			}

			((CpuKernelManager*)global_kernel_manager)
			    ->AddKernelFunction(&kernel, kernel_callback);

#ifndef NDEBUG
			cout << "Loaded kernel: " << kernel.kernel_name_ << endl;
#endif
		}
	}

	cout << "Successfully compiled and loaded kernel library." << endl;
}

}  // namespace TensorFrost
