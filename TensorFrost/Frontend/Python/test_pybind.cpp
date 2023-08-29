#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "test.cpp"

namespace py = pybind11;

PYBIND11_MODULE(test_pybind, m) {
    m.def("add_one", &add_one);
}