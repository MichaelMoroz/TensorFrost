#include <TensorFrost.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace TensorFrost {

#define PT(tensor) PyTensor(&(tensor))
#define T(tensor) (tensor).Get()

namespace py = pybind11;

void UpdateTensorNames();
class PyTensor;

using PyTensors = std::vector<PyTensor*>;

PyTensors PyTensorsFromTuple(const py::tuple& tuple);
Tensors TensorsFromTuple(const py::tuple& tuple);
PyTensors PyTensorsFromList(const py::list& list);
Tensors TensorsFromList(const py::list& list);
PyTensors PyTensorsFromTensors(const Tensors& tensors);
std::variant<PyTensor*, py::tuple> PyTensorsToTupleVariant(const PyTensors& tensors);

using ArgInfo = std::tuple<std::string, std::string, std::string>; // (name, annotation, default)

vector<ArgInfo> GetFunctionArguments(const py::function& func);

py::array ListToArray(py::list input_list);

// Tensor wrapper for python
class PyTensor {
	Tensor* tensor_;
	Tensors indices;
	Tensor* value = nullptr;

 public:
	explicit PyTensor(Tensor* tensor) : tensor_(tensor) {}
	explicit PyTensor(const Tensor* tensor) : tensor_(const_cast<Tensor*>(tensor)) {}
	~PyTensor() { UpdateTensorNames(); }

	//tensor view constructor
	explicit PyTensor(const Tensor* value, Tensors& indices)
		: value(const_cast<Tensor*>(value)), indices(std::move(indices)) {
		tensor_ = &Tensor::Load(*value, this->indices);
	}

	const Tensor& Get() const { return *tensor_; }

	Tensor* Value() const {
		if (value == nullptr) {
			throw std::runtime_error("Not a tensor view");
		}
		return value;
	}

	Tensors Indices() const { 
		if (value == nullptr) {
			throw std::runtime_error("Not a tensor view");
		}
		return indices;
	}

	explicit PyTensor(float value) { tensor_ = &Tensor::Constant(value); }
	explicit PyTensor(int value) { tensor_ = &Tensor::Constant(value); }
	explicit PyTensor(unsigned int value) { tensor_ = &Tensor::Constant(value); }
	explicit PyTensor(bool value) { tensor_ = &Tensor::Constant(value); }

	std::variant<PyTensor*, py::tuple> __enter__() {
		//py::print("Entering node scope");
		std::variant<Tensor*, Tensors> entered = tensor_->Enter();
		if (std::holds_alternative<Tensor*>(entered)) {
			return new PyTensor(std::get<Tensor*>(entered));
		} else {
			auto tensors = std::get<Tensors>(entered);
			//convert to py::tuple of PyTensor*
			return PyTensorsToTupleVariant(PyTensorsFromTensors(Reverse(tensors)));
		}
	}

	void __exit__(py::object exc_type, py::object exc_value,
	              py::object traceback) {
		//py::print("Exiting node scope");
		tensor_->Exit();
	}
};


std::string r_op(const std::string& name);
std::string l_op(const std::string& name);

}  // namespace TensorFrost