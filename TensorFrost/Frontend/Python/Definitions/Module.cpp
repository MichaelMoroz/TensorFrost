#include <utility>
#include <vector>

#include <Frontend/Python/PyModule.h>

namespace TensorFrost {

class PyModule : public Module {
public:
    using Module::Module; // Inherit constructors

    void assert_parameters() override {
        PYBIND11_OVERRIDE(void, Module, assert_parameters);
    }

    py::object loss(py::object X, py::object Y) override {
        PYBIND11_OVERRIDE_PURE(py::object, Module, loss, X, Y);
    }

    py::object forward(py::object X) override {
        PYBIND11_OVERRIDE_PURE(py::object, Module, forward, X);
    }
};

class ADAMOpt : public PyModule {
public:
    ADAMOpt(py::object net, float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : PyModule() {
        setattr("net", net);
        setattr("learning_rate", py::cast(learning_rate));
        setattr("beta1", py::cast(beta1));
        setattr("beta2", py::cast(beta2));
        setattr("epsilon", py::cast(epsilon));

        ParameterArray m, v;
        setattr("m", py::cast(m));
        setattr("v", py::cast(v));

        Parameter t({1}, TFType::Float, false);
        setattr("t", py::cast(t));

        py::list net_params = net.attr("parameters")();
        for (size_t i = 0; i < py::len(net_params); ++i) {
            py::object param = net_params[i];
            Parameter m_param = Parameter(py::cast<Parameter&>(param).shape, TFType::Float, false);
            Parameter v_param = Parameter(py::cast<Parameter&>(param).shape, TFType::Float, false);
            py::cast<ParameterArray&>(getattr("m")).setitem(i, py::cast(m_param));
            py::cast<ParameterArray&>(getattr("v")).setitem(i, py::cast(v_param));
        }
    }

    void assert_parameters() override {
        py::print("ADAMOpt::assert_parameters");
        py::list net_params = getattr("net").attr("parameters")();
        for (size_t i = 0; i < py::len(net_params); ++i) {
            py::object param = net_params[i];
            py::object m = py::cast<ParameterArray&>(getattr("m")).getitem(i);
            py::object v = py::cast<ParameterArray&>(getattr("v")).getitem(i);

            // Get the shape of the parameter tensor
            py::object shape = param.attr("shape");

            // Assert tensors
            m = tf.attr("assert_tensor")(m, shape, param.attr("type"));
            v = tf.attr("assert_tensor")(v, shape, param.attr("type"));

            py::cast<ParameterArray&>(getattr("m")).setitem(i, m);
            py::cast<ParameterArray&>(getattr("v")).setitem(i, v);
        }
    }

    py::object step(py::object X, py::object Y) {
        py::print("ADAMOpt::step");
        py::object t = getattr("t");
        t = t + py::float_(1.0);
        setattr("t", t);

        py::object net = getattr("net");
        py::object L = net.attr("loss")(X, Y);
        py::list net_params = net.attr("parameters")();

        float learning_rate = py::cast<float>(getattr("learning_rate"));
        float beta1 = py::cast<float>(getattr("beta1"));
        float beta2 = py::cast<float>(getattr("beta2"));
        float epsilon = py::cast<float>(getattr("epsilon"));

        for (size_t i = 0; i < py::len(net_params); ++i) {
            py::object param = net_params[i];
            py::object m = py::cast<ParameterArray&>(getattr("m")).getitem(i);
            py::object v = py::cast<ParameterArray&>(getattr("v")).getitem(i);

            py::object grad = tf.attr("grad")(L, param);
            grad = tf.attr("clamp")(grad, -0.1f, 0.1f);

            m = tf.attr("lerp")(m, grad, beta1);
            v = tf.attr("lerp")(v, grad.attr("__mul__")(grad), beta2);

            py::object mhat = m.attr("__truediv__")(py::float_(1.0) - tf.attr("pow")(beta1, t.attr("__getitem__")(0)));
            py::object vhat = v.attr("__truediv__")(py::float_(1.0) - tf.attr("pow")(beta2, t.attr("__getitem__")(0)));

            py::object update = mhat.attr("__truediv__")(tf.attr("sqrt")(vhat).attr("__add__")(epsilon));
            update = update.attr("__mul__")(learning_rate);
            param = param - update;

            py::cast<ParameterArray&>(getattr("v")).setitem(i, v);
            py::cast<ParameterArray&>(getattr("m")).setitem(i, m);
            net_params[i] = param;
        }

        net.attr("update_parameters")(net_params);
        return L;
    }
};

void ModuleDefinitions(py::module& m) {
    py::class_<Parameter>(m, "Parameter")
        .def(py::init<const std::vector<int>&, TFType, bool>(), py::arg("shape"), py::arg("dtype"), py::arg("random_init") = true)
        .def_readwrite("shape", &Parameter::shape)
        .def_readwrite("dtype", &Parameter::dtype)
        .def_readwrite("random_init", &Parameter::random_init)
        .def("__repr__", [](const Parameter& p) {
            return "Parameter(shape=" + std::to_string(p.shape.size()) + ", dtype=" + std::to_string(p.dtype) + ", random_init=" + std::to_string(p.random_init) + ")";
        });

    py::class_<ParameterArray>(m, "ParameterArray")
        .def(py::init<>())
        .def("__getitem__", &ParameterArray::getitem)
        .def("__setitem__", &ParameterArray::setitem);

    py::class_<Module, PyModule>(m, "Module")
        .def(py::init<>())
        .def("__getattr__", &Module::getattr)
        .def("__setattr__", &Module::setattr)
        .def("initialize_input", &Module::initialize_input)
        .def("initialize_parameters", &Module::initialize_parameters)
        .def("parameters", &Module::parameters)
        .def("create_input", &Module::create_input)
        .def("update_parameters", &Module::update_parameters)
        .def("assert_parameters", &Module::assert_parameters)
        .def("loss", &Module::loss)
        .def("forward", &Module::forward);

    // Create a nested module for optimizers
    py::module optimizers = m.def_submodule("optimizers", "Optimizers submodule");

    // Define ADAMOpt in the optimizers submodule
    py::class_<ADAMOpt, Module>(optimizers, "adam")
        .def(py::init<py::object, float, float, float, float>(),
             py::arg("net"), py::arg("learning_rate"), py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("epsilon") = 1e-8f)
        .def("assert_parameters", &ADAMOpt::assert_parameters)
        .def("step", &ADAMOpt::step);
}

}  // namespace TensorFrost