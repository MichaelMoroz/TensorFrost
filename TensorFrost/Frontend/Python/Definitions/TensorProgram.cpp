#include <utility>
#include <vector>

#include <Frontend/Python/PyTensor.h>
#include <Frontend/Python/PyTensorMemory.h>
#include <Frontend/Python/PyModule.h>

namespace TensorFrost {

void TensorProgramDefinition(py::module& m,
                             py::class_<TensorProgram>& tensor_program) {
	m.def(
	    "compile",
	    [](const py::function& py_evaluate) {
		    // Extract the name of the Python function
		    std::string func_name =
		        py_evaluate.attr("__name__").cast<std::string>();

	    	vector<ArgInfo> input_names = GetFunctionArguments(py_evaluate);

		    TensorProgram& program = *new TensorProgram(
		        [py_evaluate]() -> Tensors {
			        py::gil_scoped_acquire acquire;
			        py::object result = py_evaluate();
					Tensors outputs;
					//if the result is a single tensor
		        	if (py::isinstance<PyTensor>(result)) {
		        		outputs.push_back(&py::cast<PyTensor&>(result).Get());
		        	} else {
		        		auto py_outputs = py::cast<vector<PyTensor>>(result);
						for (PyTensor output : py_outputs) {
							outputs.push_back(&output.Get());
						}
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
	    [](TensorProgram& program, py::args py_inputs) -> std::variant<py::object, py::tuple> {
	    	vector<py::object> inputs_props;
	    	vector<TFTensor*> temp_numpy_tensors;
			for (auto arg : py_inputs) {
				if (py::isinstance<PyTensorMemory>(arg)) { //if just tensor memory
					PyTensorMemory* mem = &arg.cast<PyTensorMemory&>();
					inputs_props.push_back(arg.cast<py::object>());
				} else if (py::isinstance<Module>(arg)) { //if module then add its parameters
					Module* module = &arg.cast<Module&>();
					py::list params = module->parameters();
					for (auto param : params) {
						inputs_props.push_back(param.cast<py::object>());
					}
				} else if (py::isinstance<py::array>(arg)) { //if numpy array then create pytensormemory from it and add it
					py::array arr = arg.cast<py::array>();
					PyTensorMemory* temp_tensor = new PyTensorMemory(arr);
					inputs_props.push_back(py::cast(temp_tensor, py::return_value_policy::take_ownership));
					temp_numpy_tensors.push_back(temp_tensor->tensor_);
				} else if (py::isinstance<py::list>(arg)) { //if list then convert to py::array then create pytensormemory from it and add it
					py::array arr = ListToArray(arg.cast<py::list>());
					PyTensorMemory* temp_tensor = new PyTensorMemory(arr);
					inputs_props.push_back(py::cast(temp_tensor, py::return_value_policy::take_ownership));
					temp_numpy_tensors.push_back(temp_tensor->tensor_);
				} else {
					throw std::runtime_error("Unsupported input type " + std::string(py::str(arg)));
				}
			}

	    	vector<TFTensor*> inputs;
	    	for (auto input : inputs_props) {
	    		PyTensorMemory* mem = input.cast<PyTensorMemory*>();
	    		inputs.push_back(mem->tensor_);
	    	}
		    vector<TFTensor*> outputs = program.Evaluate(inputs);

	    	//remove temporary tensors if they are not in the outputs
	    	for (TFTensor* temp_tensor : temp_numpy_tensors) {
	    		bool found = false;
	    		for (TFTensor* output : outputs) {
	    			if (temp_tensor->buffer == output->buffer) {
	    				found = true;
	    				break;
	    			}
	    		}
	    		if (!found) {
	    			global_memory_manager->DeallocateTensor(*temp_tensor);
	    		}
	    	}

	    	vector<py::object> output_tensors;
	    	for (size_t i = 0; i < outputs.size(); i++) {
	    		//if any of the outputs are also inputs, then replace them with the input tensors
	    		TFTensor* out = outputs[i];
	    		bool is_input = false;
	    		for (size_t j = 0; j < inputs_props.size(); j++) {
					PyTensorMemory* in = inputs_props[j].cast<PyTensorMemory*>();
					if (out->buffer == in->tensor_->buffer) {
						output_tensors.push_back(inputs_props[j]);
						is_input = true;
						break;
					}
				}
	    		if (is_input) {
	    			continue;
	    		}
	    		//otherwise create a new tensor memory
	    		output_tensors.push_back(py::cast(new PyTensorMemory(outputs[i]), py::return_value_policy::take_ownership));
	    	}

	    	//if there is only one output, return the tensor memory
	    	if (outputs.size() == 1) {
	    		return output_tensors[0];
	    	} else {
	    		//convert to py::tuple of PyTensorMemory*
	    		py::tuple py_outputs = py::tuple(outputs.size());
	    		for (size_t i = 0; i < outputs.size(); i++) {
	    			py_outputs[i] = output_tensors[i];
	    		}
	    		return py_outputs;
	    	}
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
			kernel_source.push_back(kernel.full_generated_code_);
		}
		return kernel_source;
	});

	tensor_program.def("get_main_function", [](TensorProgram& program) {
		return program.program->main_function_;
	});

	m.def("get_all_generated_main_functions", []() {
		return global_kernel_manager->GetAllMainFunctions();
	});

	m.def("get_all_generated_kernels", []() {
		return global_kernel_manager->GetAllKernels();
	});

	m.def("get_cpp_header", []() {
		return GetCPPHeader();
	});

	m.def("get_cpp_implementation", []() {
		return GetCPPImplementation();
	});
}

}  // namespace TensorFrost
