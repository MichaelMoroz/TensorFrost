#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

#include "../../Tensor/Tensor.h"

std::vector<int> PyArrayToVector(py::array_t<float> array)
{
    auto buffer = array.request();
    //check if only 1 dimension
    if (buffer.ndim != 1)
    {
        throw std::runtime_error("Only 1D arrays are supported");
    }
    float *ptr = (float *) buffer.ptr;
    std::vector<int> vec = std::vector<int>();
    for (int i = 0; i < buffer.shape[0]; i++)
    {
        vec.push_back(ptr[i]);
    }
    return vec;
}

Tensor TensorFromPyArray(py::array_t<float> array)
{
    auto buffer = array.request();
    float *ptr = (float *) buffer.ptr;
    std::vector<int> shape = std::vector<int>();
    for (int i = 0; i < buffer.ndim; i++)
    {
        shape.push_back(buffer.shape[i]);
    }
    return Tensor(ptr, shape);
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

    #define DEFINE_OPERATOR(opname, op) \
        .def("__" #opname "__", [](const Tensor& t, const Tensor& t2) { return t op t2; }) \
        .def("__" #opname "__", [](const Tensor& t, float f) { return t op (Tensor)f; }) \
        .def("__" #opname "__", [](const Tensor& t, py::array_t<float> f) { return t op TensorFromPyArray(f); }) \
        .def("__r" #opname "__", [](const Tensor& t, float f) { return (Tensor)f op t; }) \
        .def("__r" #opname "__", [](const Tensor& t, py::array_t<float> f) { return TensorFromPyArray(f) op t; })

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<int>>())
        .def("get", &Tensor::get)
        .def("set", &Tensor::set)
        .def("__getitem__", &Tensor::get)
        .def("__setitem__", &Tensor::set)
        .def_property_readonly("shape", [] (const Tensor& t) { return py::array_t<int>(t.shape.size(), t.shape.data()); })
        .def_property_readonly("type", [] (const Tensor& t) { return t.type; })
        .def("numpy", [](const Tensor& t) { return TensorToPyArray(t); })
        //operator overloads
        DEFINE_OPERATOR(add, +)
        DEFINE_OPERATOR(sub, -)
        DEFINE_OPERATOR(mul, *)
        DEFINE_OPERATOR(div, /)
        DEFINE_OPERATOR(mod, %)
        //negative
        .def("__neg__", [](const Tensor& t) { return -t; })
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
        .def("__not__", [](const Tensor& t) { return !t; })
        //bitwise
        DEFINE_OPERATOR(xor, ^)
        DEFINE_OPERATOR(lshift, <<)
        DEFINE_OPERATOR(rshift, >>)
        DEFINE_OPERATOR(and_, &)
        DEFINE_OPERATOR(or_, |)
        .def("__invert__", [](const Tensor& t) { return ~t; })
        //end operator overloads
        ;
 
    //unary functions
    #define UNARY_FUNCTION(name) \
        m.def(#name, [](const Tensor& t) { return Tensor::name(t); })

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
        m.def(#name, [](const Tensor& t, const Tensor& t2) { return Tensor::name(t, t2); }) \
        .def(#name, [](const Tensor& t, float f) { return Tensor::name(t, (Tensor)f); }) \
        .def(#name, [](const Tensor& t, py::array_t<float> f) { return Tensor::name(t, TensorFromPyArray(f)); }) \
        .def(#name, [](float f, const Tensor& t) { return Tensor::name((Tensor)f, t); }) \
        .def(#name, [](py::array_t<float> f, const Tensor& t) { return Tensor::name(TensorFromPyArray(f), t); })
    
    //basic
    BINARY_FUNCTION(min);
    BINARY_FUNCTION(max);
    BINARY_FUNCTION(pow);
    BINARY_FUNCTION(atan2);
    
    //end binary functions

    //ternary functions
    #define TERNARY_FUNCTION(name) \
        m.def(#name, [](const Tensor& t, const Tensor& t2, const Tensor& t3) { return Tensor::name(t, t2, t3); }) \
        .def(#name, [](const Tensor& t, const Tensor& t2, float f) { return Tensor::name(t, t2, (Tensor)f); }) \
        .def(#name, [](const Tensor& t, const Tensor& t2, py::array_t<float> f) { return Tensor::name(t, t2, TensorFromPyArray(f)); }) \
        .def(#name, [](const Tensor& t, float f, const Tensor& t2) { return Tensor::name(t, (Tensor)f, t2); }) \
        .def(#name, [](const Tensor& t, py::array_t<float> f, const Tensor& t2) { return Tensor::name(t, TensorFromPyArray(f), t2); }) \
        .def(#name, [](float f, const Tensor& t, const Tensor& t2) { return Tensor::name((Tensor)f, t, t2); }) \
        .def(#name, [](py::array_t<float> f, const Tensor& t, const Tensor& t2) { return Tensor::name(TensorFromPyArray(f), t, t2); })

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
    


    


}
