#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <vector>

namespace py = pybind11;

#include "../../Tensor/Tensor.h"
#include "../../Tensor/TensorProgram.h"

namespace TensorFrost
{

//Tensor wrapper for python
class PyTensor
{
    Tensor* tensor;

public:
    PyTensor(Tensor* tensor) : tensor(tensor) {}
    ~PyTensor() {}

    Tensor& Get() const
    { 
        return *tensor; 
    }

    PyTensor(std::vector<int> shape, DataType type)
    {
        tensor = &Tensor::Constant(shape, 0.0f);
    }
};


Tensor TensorFromPyArray(py::array_t<float> array)
{
    auto buffer = array.request();
    float *ptr = (float *) buffer.ptr;
    std::vector<int> shape = std::vector<int>();
    for (int i = 0; i < buffer.ndim; i++)
    {
        shape.push_back(buffer.shape[i]);
    }
    return Tensor::Constant(shape, ptr);
}

py::array_t<float> TensorToPyArray(const Tensor& tensor)
{
    std::vector<int> shape = tensor.shape;
    py::array::ShapeContainer shape2 = py::array::ShapeContainer(shape.begin(), shape.end());
    py::array_t<float> array(shape2);
    auto buffer = array.request();
    float *ptr = (float *) buffer.ptr;
    for (int i = 0; i < tensor.Size(); i++)
    {
        ptr[i] = 0.0;
    }
    return array;
}

PYBIND11_MODULE(TensorFrost, m) 
{
    py::enum_<DataType>(m, "DataType")
        .value("f32", DataType::f32)
        .value("i32", DataType::i32)
        .value("u32", DataType::u32)
        .value("b1", DataType::b1);

    #define PT(tensor) PyTensor(&(tensor))
    #define T(tensor) (tensor).Get()

    #define DEFINE_OPERATOR(opname, op) \
        .def("__" #opname "__", [](const PyTensor& t, const PyTensor& t2) { return PT(T(t) op T(t2)); }) \
        .def("__" #opname "__", [](const PyTensor& t, float f) { return PT(T(t) op Tensor::Constant(T(t).shape, f)); }) \
        .def("__" #opname "__", [](const PyTensor& t, py::array_t<float> f) { return PT(T(t) op TensorFromPyArray(f)); }) \
        .def("__r" #opname "__", [](const PyTensor& t, float f) { return PT(Tensor::Constant(T(t).shape, f) op T(t)); }) \
        .def("__r" #opname "__", [](const PyTensor& t, py::array_t<float> f) { return PT(TensorFromPyArray(f) op T(t)); }) \

    py::class_<PyTensor>(m, "Tensor")
        .def(py::init<std::vector<int>, DataType>())
        .def("get", [](const PyTensor& t, std::vector<int> indices) { return t.Get().get(indices); })
        .def("set", [](const PyTensor& t, std::vector<int> indices, float value) { t.Get().set(indices, value); })
        .def("__getitem__", [](const PyTensor& t, std::vector<int> indices) { return t.Get().get(indices); })
        .def("__setitem__", [](const PyTensor& t, std::vector<int> indices, float value) { t.Get().set(indices, value); })
        .def_property_readonly("shape", [] (const PyTensor& t) { return t.Get().shape; })
        .def_property_readonly("type", [] (const PyTensor& t) { return t.Get().type; })
        .def("numpy", [](const PyTensor& t) { return TensorToPyArray(t.Get()); })
        //operator overloads
        DEFINE_OPERATOR(add, +)
        DEFINE_OPERATOR(sub, -)
        DEFINE_OPERATOR(mul, *)
        DEFINE_OPERATOR(div, /)
        DEFINE_OPERATOR(mod, %)
        //negative
        .def("__neg__", [](const PyTensor& t) { return PT(-T(t)); })
        //comparison
        DEFINE_OPERATOR(eq, ==)
        DEFINE_OPERATOR(ne, !=)
        DEFINE_OPERATOR(lt, <)
        DEFINE_OPERATOR(le, <=)
        DEFINE_OPERATOR(gt, >)
        DEFINE_OPERATOR(ge, >=)
        //logical
        DEFINE_OPERATOR(and, &&)
        DEFINE_OPERATOR(or, ||)
        .def("__not__", [](const PyTensor& t) { return PT(!T(t)); })
        //bitwise
        DEFINE_OPERATOR(xor, ^)
        DEFINE_OPERATOR(lshift, <<)
        DEFINE_OPERATOR(rshift, >>)
        DEFINE_OPERATOR(and_, &)
        DEFINE_OPERATOR(or_, |)
        .def("__invert__", [](const PyTensor& t) { return PT(~T(t)); })
        //power operator
        .def("__pow__", [](const PyTensor& t, const PyTensor& t2) { return PT(Tensor::pow(T(t), T(t2))); })
        .def("__pow__", [](const PyTensor& t, float f) { return PT(Tensor::pow(T(t), Tensor::Constant(T(t).shape, f))); })
        .def("__pow__", [](const PyTensor& t, py::array_t<float> f) { return PT(Tensor::pow(T(t), TensorFromPyArray(f))); })
        .def("__rpow__", [](const PyTensor& t, float f) { return PT(Tensor::pow(Tensor::Constant(T(t).shape, f), T(t))); })
        .def("__rpow__", [](const PyTensor& t, py::array_t<float> f) { return PT(Tensor::pow(TensorFromPyArray(f), T(t))); })
        //end power operator
        //end operator overloads
        ;
 
    //unary functions
    #define UNARY_FUNCTION(name) \
        m.def(#name, [](const PyTensor& t) { return PT(Tensor::name(T(t))); }) \

    //basic
    UNARY_FUNCTION(abs);
    UNARY_FUNCTION(ceil);
    UNARY_FUNCTION(floor);
    UNARY_FUNCTION(round);
    UNARY_FUNCTION(trunc);
    UNARY_FUNCTION(sign);
    UNARY_FUNCTION(frac);

    //trigonometric
    UNARY_FUNCTION(sin);
    UNARY_FUNCTION(cos);
    UNARY_FUNCTION(tan);
    UNARY_FUNCTION(asin);
    UNARY_FUNCTION(acos);
    UNARY_FUNCTION(atan);
    UNARY_FUNCTION(sinh);
    UNARY_FUNCTION(cosh);
    UNARY_FUNCTION(tanh);

    //exponential
    UNARY_FUNCTION(exp);
    UNARY_FUNCTION(exp2);
    UNARY_FUNCTION(log);
    UNARY_FUNCTION(log2);
    UNARY_FUNCTION(sqrt);
    UNARY_FUNCTION(sqr);
    UNARY_FUNCTION(rsqrt);
    UNARY_FUNCTION(rcp);
    
    //end unary functions

    //binary functions
    #define BINARY_FUNCTION(name) \
        m.def(#name, [](const PyTensor& t, const PyTensor& t2) { return PT(Tensor::name(T(t), T(t2))); }) \
        .def(#name, [](const PyTensor& t, float f) { return PT(Tensor::name(T(t), Tensor::Constant(T(t).shape, f))); }) \
        .def(#name, [](const PyTensor& t, py::array_t<float> f) { return PT(Tensor::name(T(t), TensorFromPyArray(f))); }) \
        .def(#name, [](float f, const PyTensor& t) { return PT(Tensor::name(Tensor::Constant(T(t).shape, f), T(t))); }) \
        .def(#name, [](py::array_t<float> f, const PyTensor& t) { return PT(Tensor::name(TensorFromPyArray(f), T(t))); }) \
    
    //basic
    BINARY_FUNCTION(min);
    BINARY_FUNCTION(max);
    BINARY_FUNCTION(pow);
    BINARY_FUNCTION(atan2);
    
    //end binary functions

    //ternary functions
    #define TERNARY_FUNCTION(name) \
        m.def(#name, [](const PyTensor& t, const PyTensor& t2, const PyTensor& t3) { return PT(Tensor::name(T(t), T(t2), T(t3))); }) \
        .def(#name, [](const PyTensor& t, float f, const PyTensor& t2) { return PT(Tensor::name(T(t), Tensor::Constant(T(t).shape, f), T(t2))); }) \
        .def(#name, [](const PyTensor& t, py::array_t<float> f, const PyTensor& t2) { return PT(Tensor::name(T(t), TensorFromPyArray(f), T(t2))); }) \
        .def(#name, [](const PyTensor& t, const PyTensor& t2, float f) { return PT(Tensor::name(T(t), T(t2), Tensor::Constant(T(t).shape, f))); }) \
        .def(#name, [](const PyTensor& t, const PyTensor& t2, py::array_t<float> f) { return PT(Tensor::name(T(t), T(t2), TensorFromPyArray(f))); }) \
        .def(#name, [](float f, const PyTensor& t, const PyTensor& t2) { return PT(Tensor::name(Tensor::Constant(T(t).shape, f), T(t), T(t2))); }) \
        .def(#name, [](py::array_t<float> f, const PyTensor& t, const PyTensor& t2) { return PT(Tensor::name(TensorFromPyArray(f), T(t), T(t2))); }) \

    //basic
    TERNARY_FUNCTION(clamp);
    TERNARY_FUNCTION(fma);
    TERNARY_FUNCTION(lerp);
    
    //end ternary functions

    //special functions
    //m.def("thread_id", [](py::array_t<float> shape) { return Tensor::thread_id(PyArrayToVector(shape)); });
    //m.def("dim", [](int dim, py::array_t<float> shape) { return Tensor::dim(dim, PyArrayToVector(shape)); });
    //m.def("rand", [](py::array_t<float> shape) { return Tensor::rand(PyArrayToVector(shape)); });
    //m.def("randn", [](py::array_t<float> shape) { return Tensor::randn(PyArrayToVector(shape)); });
    //m.def("break", []() { return Tensor::break_(); });
    //m.def("continue", []() { return Tensor::continue_(); });
    //m.def("loop", [](int count, const py::function& func) { return Tensor::loop(count, func); });
//
    //py::class_<IndexedTensor>(m, "IndexedTensor")
    //    .def(py::init<&Tensor, std::vector<&Tensor>>())
    //    ;
//
    ////scatter functions
    //m.def("scatterAdd", [](const Tensor& t, const Tensor& t2, const Tensor& t3) { return Tensor::scatterAdd(t, t2, t3); });
    //m.def("scatterMin", [](const Tensor& t, const Tensor& t2, const Tensor& t3) { return Tensor::scatterMin(t, t2, t3); });
    //m.def("scatterMax", [](const Tensor& t, const Tensor& t2, const Tensor& t3) { return Tensor::scatterMax(t, t2, t3); });
    
    m.def("zeros", [](std::vector<int> shape) { 
        std::string debug = "Received shape: " + std::to_string(shape[0]);
        for (int i = 1; i < shape.size(); i++)
        {
            debug += ", " + std::to_string(shape[i]);
        }
        py::print(debug);
		return PT(Tensor::Constant(shape, 0.0f));
    });

    m.def("Program", [](const py::function& py_evaluate) {
        return TensorProgram([py_evaluate](std::vector<Tensor> inputs) -> std::vector<Tensor> {
            py::gil_scoped_acquire acquire; // Acquire the GIL

            // Create C++ vector of PyTensors
            std::vector<PyTensor> py_in;
            for (Tensor input : inputs)
            {
                py_in.push_back(PT(input));
            }

            // 1. Convert C++ vector to Python list
            py::list py_inputs = py::cast(py_in);

            // Debug print to ensure conversion was successful
            py::print("Converted to py_inputs:", py_inputs);

            // 2. Call the Python function
            py::object result = py_evaluate(py_inputs);

            // Debug print to check the result
            py::print("Result from Python function:", result);

            // 3. Convert back to std::vector<Tensor>
            std::vector<PyTensor> py_outputs = py::cast<std::vector<PyTensor>>(result);
            std::vector<Tensor> outputs = std::vector<Tensor>();
            for (PyTensor output : py_outputs)
            {
                outputs.push_back(output.Get());
            }

            return outputs;
        });
    }, "Compile a TensorProgram from a python function");

    py::class_<TensorProgram>(m, "TensorProgram")
        .def("__call__", [](TensorProgram& program, py::list py_inputs) {
            std::vector<PyTensor> inputs = py::cast<std::vector<PyTensor>>(py_inputs);
            py::print("Received inputs: " + std::to_string(inputs.size()));
            std::vector<Tensor> inputs2 = std::vector<Tensor>();
            for (PyTensor input : inputs)
            {
                inputs2.push_back(input.Get());
            }
            std::vector<Tensor> outputs = program.Evaluate(inputs2);
            std::vector<PyTensor> outputs2 = std::vector<PyTensor>();
            for (Tensor output : outputs)
            {
                outputs2.push_back(PT(output));
            }
            py::print("Returning outputs: " + std::to_string(outputs2.size()));
            return py::cast(outputs2);
        }, "Evaluate the TensorProgram with the given inputs")
        .def("ListGraphOperations", [](TensorProgram& program) {
            std::string listing = "List of operations:\n";
            listing += program.ir.GetOperationListing();
            py::str result = py::str(listing);
            py::print(result);
            });
        ;

    py::print("TensorFrost module loaded!");
}

}