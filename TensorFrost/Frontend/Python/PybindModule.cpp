#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>
#include <Frontend/Python/PyTensorMemory.h>
#include <pybind11/eval.h>

namespace TensorFrost {

void PyTensorDefinition(py::module&, py::class_<PyTensor>&);
void TensorFunctionsDefinition(py::module&);
void TensorProgramDefinition(py::module&, py::class_<TensorProgram>&);
void TensorMemoryDefinition(py::module& m,
                            py::class_<PyTensorMemory>& py_tensor_mem);
void WindowDefinitions(py::module& m);
void ScopeDefinitions(py::module& m, py::class_<PyTensor>& py_tensor);
void ModuleDefinitions(py::module& m);

PYBIND11_MODULE(TensorFrost, m) {
	m.doc() = "TensorFrost library";
	auto data_type = py::enum_<TFType>(m, "TFType");
	auto backend_type = py::enum_<BackendType>(m, "BackendType");
	auto code_gen_lang = py::enum_<CodeGenLang>(m, "CodeGenLang");
	auto py_tensor = py::class_<PyTensor>(m, "Tensor");
	auto tensor_program = py::class_<TensorProgram>(m, "TensorProgram");
	auto py_tensor_mem = py::class_<PyTensorMemory>(m, "TensorMemory");
	auto py_tensor_arg = py::class_<PyTensorArg>(m, "Arg");

	data_type.value("float", TFType::Float);
	data_type.value("int", TFType::Int);
	data_type.value("uint", TFType::Uint);
	data_type.value("bool", TFType::Bool);

	auto data_format = py::class_<TFDataFormat>(m, "TFDataFormat");
	data_format.def(py::init<TFType, size_t>());
	data_format.def_readwrite("type", &TFDataFormat::type);
	data_format.def_readwrite("size", &TFDataFormat::size);
	// Add printers for the enums
	data_format.def("__repr__", [](const TFDataFormat& a) {
		return "<TFDataFormat: type=" + DataTypeToString(a.type) + ", size=" + std::to_string(a.size) + ">";
	});
	data_format.def("__str__", [](const TFDataFormat& a) {
		return "<TFDataFormat: type=" + DataTypeToString(a.type) + ", size=" + std::to_string(a.size) + ">";
	});
	data_type.def("__repr__", [](TFType a) {
		return "<TFType: " + DataTypeToString(a) + ">";
	});
	data_type.def("__str__", [](TFType a) {
		return "<TFType: " + DataTypeToString(a) + ">";
	});

	backend_type.value("cpu", BackendType::CPU);
	backend_type.value("vulkan", BackendType::Vulkan);
	backend_type.value("opengl", BackendType::OpenGL);
	backend_type.value("codegen", BackendType::CodeGen);
	code_gen_lang.value("cpp", CodeGenLang::CPP);
	code_gen_lang.value("glsl", CodeGenLang::GLSL);
	code_gen_lang.value("hlsl", CodeGenLang::HLSL);

	data_format.def(py::self == py::self);
	backend_type.def("__eq__", [](BackendType a, BackendType b) {
		return a == b;
	});
	code_gen_lang.def("__eq__", [](CodeGenLang a, CodeGenLang b) {
		return a == b;
	});
	data_type.def("__eq__", [](TFType a, TFType b) {
		return a == b;
	});

	m.attr("float32") = TFTypeFloat32;
	m.attr("int32") = TFTypeInt32;
	m.attr("uint32") = TFTypeUint32;
	m.attr("bool1") = TFTypeBool32;

	m.attr("cpu") = BackendType::CPU;
	m.attr("vulkan") = BackendType::Vulkan;
	m.attr("opengl") = BackendType::OpenGL;
	m.attr("codegen") = BackendType::CodeGen;

	m.attr("cpp_lang") = CodeGenLang::CPP;
	m.attr("glsl_lang") = CodeGenLang::GLSL;
	m.attr("hlsl_lang") = CodeGenLang::HLSL;

	PyTensorDefinition(m, py_tensor);

	// implicit conversion from TensorView to PyTensor
	py::implicitly_convertible<float, PyTensor>();
	py::implicitly_convertible<bool, PyTensor>();
	py::implicitly_convertible<int, PyTensor>();
	py::implicitly_convertible<unsigned int, PyTensor>();

	TensorFunctionsDefinition(m);
	TensorProgramDefinition(m, tensor_program);
	TensorMemoryDefinition(m, py_tensor_mem);
	WindowDefinitions(m);
	ScopeDefinitions(m, py_tensor);
	ModuleDefinitions(m);

	py_tensor_arg.def(py::init([](py::list shape, TFDataFormat type) {
		std::vector<int> shape_vec;
		for (auto& s : shape) {
			shape_vec.push_back(s.cast<int>());
		}
		return PyTensorArg(shape_vec, type);
	}), "Create a TensorArg with the given shape and type", py::return_value_policy::take_ownership);

	m.def("current_backend", []() {
		return current_backend;
	}, "Get the current backend");

	m.def("initialize",
	      [](BackendType backend_type, const std::string& kernel_compile_options, CodeGenLang kernel_lang) {
		      InitializeBackend(backend_type, kernel_compile_options, kernel_lang);
	      }, py::arg("backend_type") = BackendType::CPU, py::arg("kernel_compile_options") = "", py::arg("kernel_lang") = CodeGenLang::None, "Initialize the backend");

	m.def("strip_debug_info", [](bool strip) {
		strip_debug_names = strip;
	}, py::arg("strip") = true, "Strip debug info from the kernel");

#ifdef NDEBUG
	py::print("TensorFrost module loaded!");
#else
	py::print("TensorFrost module loaded in debug mode! Expect slow performance.");
#endif
}

}  // namespace TensorFrost