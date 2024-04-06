#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

void TensorProgramDefinition(py::module& m,
                             py::class_<TensorProgram>& tensor_program) {
	m.def(
	    "compile",
	    [](const py::function& py_evaluate) {
		    // Extract the name of the Python function
		    std::string func_name =
		        py_evaluate.attr("__name__").cast<std::string>();

		    TensorProgram& program = *new TensorProgram(
		        [py_evaluate]() -> Tensors {
			        py::gil_scoped_acquire acquire;
			        py::object result = py_evaluate();
			        auto py_outputs = py::cast<vector<PyTensor>>(result);
			        Tensors outputs = Tensors();
			        for (PyTensor output : py_outputs) {
				        outputs.push_back(&output.Get());
			        }
			        return outputs;
		        },
		        func_name);
		    
		    py::print(program.PrintProperties());
			return &program;
	    },
	    "Compile a TensorProgram from a python function");

	tensor_program.def(
	    "__call__",
	    [](TensorProgram& program, py::args py_inputs) {
		    vector<TensorMemory*> inputs = TensorMemoryFromTuple(py_inputs);
		    vector<TensorMemory*> outputs = program.Evaluate(inputs);
		    // output a tuple of tensor memories
		    py::tuple py_outputs = py::tuple(outputs.size());
		    for (size_t i = 0; i < outputs.size(); i++) {
			    py_outputs[i] = py::cast(*outputs[i]);
		    }
		    return py_outputs;
	    },
	    "Evaluate the TensorProgram with the given inputs");

	tensor_program.def(
	    "__call__",
	    [](TensorProgram& program, py::tuple py_inputs) {
		    vector<TensorMemory*> inputs = TensorMemoryFromTuple(py_inputs);
		    vector<TensorMemory*> outputs = program.Evaluate(inputs);
		    // output a tuple of tensor memories
		    py::tuple py_outputs = py::tuple(outputs.size());
		    for (size_t i = 0; i < outputs.size(); i++) {
			    py_outputs[i] = py::cast(*outputs[i]);
		    }
		    return py_outputs;
	    },
	    "Evaluate the TensorProgram with the given inputs");

	tensor_program.def(
	    "list_operations",
	    [](TensorProgram& program, bool compact) {
		    std::string listing = "List of operations:\n";
		    listing += GetOperationListing(program.ir, compact);
		    return py::str(listing);
	    },
	    py::arg("compact") = true);

	tensor_program.def("compiled_code", [](TensorProgram& program) {
		string code = program.program->generated_code_;
		return py::str(code);
	});

	tensor_program.def("get_kernels", [](TensorProgram& program) {
		vector<string> kernel_source;
		for (auto& kernel : program.program->kernels_) {
			kernel_source.push_back(kernel.generated_code_);
		}
		return kernel_source;
	});
}

}  // namespace TensorFrost
