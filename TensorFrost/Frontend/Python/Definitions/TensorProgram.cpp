#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>

namespace TensorFrost {

void TensorProgramDefinition(py::module& m,
                             py::class_<TensorProgram>& tensorProgram) {
	m.def(
	    "Program",
	    [](const py::function& py_evaluate) {
		    return TensorProgram([py_evaluate]() -> std::vector<Tensor> {
			    py::gil_scoped_acquire acquire;  // Acquire the GIL

			    // 2. Call the Python function
			    py::object result = py_evaluate();

			    // 3. Convert back to std::vector<Tensor>
			    auto py_outputs = py::cast<std::vector<PyTensor>>(result);
			    std::vector<Tensor> outputs = std::vector<Tensor>();
			    for (PyTensor output : py_outputs) {
				    outputs.push_back(output.Get());
			    }

			    return outputs;
		    });
	    },
	    "Compile a TensorProgram from a python function");

	tensorProgram.def(
	    "__call__",
	    [](TensorProgram& /*program*/, py::list py_inputs) {
		    auto inputs = py::cast<std::vector<PyTensor>>(py_inputs);
		    std::vector<Tensor> inputs2 = std::vector<Tensor>();
		    for (PyTensor input : inputs) {
			    inputs2.push_back(input.Get());
		    }
		    std::vector<Tensor> outputs =
		        TensorFrost::TensorProgram::Evaluate(inputs2);
		    std::vector<PyTensor> outputs2 = std::vector<PyTensor>();
		    for (Tensor output : outputs) {
			    outputs2.push_back(PT(output));
		    }
		    return py::cast(outputs2);
	    },
	    "Evaluate the TensorProgram with the given inputs");
	tensorProgram.def("ListGraphOperations", [](TensorProgram& program) {
		std::string listing = "List of operations:\n";
		listing += program.ir->GetOperationListing();
		py::str result = py::str(listing);
		py::print(result);
	});
}

}  // namespace TensorFrost
