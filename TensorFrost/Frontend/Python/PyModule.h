#include <TensorFrost.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace TensorFrost {

namespace py = pybind11;

class Parameter {
public:
    std::vector<int> shape;
    TFType dtype;
    bool random_init;

    Parameter(const std::vector<int>& shape, TFType dtype, bool random_init = true)
        : shape(shape), dtype(dtype), random_init(random_init) {}
};

class ParameterArray {
public:
    std::string _prefix;
    py::dict _parameters;

    ParameterArray(std::string prefix) : _prefix(prefix) {
        _parameters = py::dict();
    }

    py::object __getitem__(int index) {
        std::string key = _prefix + "_" + std::to_string(index);
        if (_parameters.contains(key)) {
            return _parameters.attr("__getitem__")(key);
        }
        throw py::index_error("Parameter '" + key + "' not found");
    }

    void __setitem__(int index, py::object value) {
        std::string key = _prefix + "_" + std::to_string(index);
        _parameters.attr("__setitem__")(key, value);
    }
};

class Module {
public:
    py::dict _parameters;
    py::dict _modules;
    py::dict _parameter_arrays;

    py::dict __dict__;

    py::object tf;

    Module() {
        tf = py::module::import("TensorFrost");
        _parameters = py::dict();
        _modules = py::dict();
        _parameter_arrays = py::dict();
        __dict__ = py::dict();
    }

    void register_parameter(const std::string& name, py::object param) {
        _parameters.attr("__setitem__")(name, param);
    }

    void register_module(const std::string& name, py::object module) {
        _modules.attr("__setitem__")(name, module);
    }

    void register_parameter_array(const std::string& name) {
        py::object array = py::cast(ParameterArray(name));
        _parameter_arrays.attr("__setitem__")(name, array);
    }

    static const char* class_name() {
        return "Module";
    }

    py::object __getattr__(const std::string& name) {
        if (_parameter_arrays.contains(name)) {
            return _parameter_arrays.attr("__getitem__")(name);
        } else if (_modules.contains(name)) {
            return _modules.attr("__getitem__")(name);
        } else if (__dict__.contains(name)) {
            return __dict__[name.c_str()];
        } else if (_parameters.contains(name)) {
            return _parameters.attr("__getitem__")(name);
        }

        throw py::attribute_error("'" + std::string(class_name()) + "' object has no attribute '" + name + "'");
    }

    void __setattr__(const std::string& name, py::object value) {
        if (py::isinstance<Parameter>(value)) {
            _parameters.attr("__setitem__")(name, value);
        } else if (py::isinstance<Module>(value)) {
            _modules.attr("__setitem__")(name, value);
        } else if (py::isinstance<ParameterArray>(value)) {
            _parameter_arrays.attr("__setitem__")(name, value);
        } else {
            // Set as a regular Python attribute
            __dict__[name.c_str()] = value;
        }
    }

    virtual void assert_parameters() {}

    void initialize_input() {
        for (auto& module : _modules) {
            module.second.attr("initialize_input")();
        }

        for (auto& param : _parameters) {
            py::object tensor = tf.attr("input")(param.second.attr("shape"), param.second.attr("dtype"));
            _parameters.attr("__setitem__")(param.first, tensor);
        }

        for (auto& array : _parameter_arrays) {
            ParameterArray& param_array = py::cast<ParameterArray&>(array.second);
            for (auto& param : param_array._parameters) {
                py::object tensor = tf.attr("input")(param.second.attr("shape"), param.second.attr("dtype"));
                param_array._parameters.attr("__setitem__")(param.first, tensor);
            }
        }

        assert_parameters();
    }

    py::object initialize_parameter(Parameter& param) {
        if (param.random_init) {
            py::object np = py::module::import("numpy");
            py::object random = np.attr("random");

            // Convert the shape vector to a tuple
            py::tuple shape_tuple = py::cast(param.shape);

            py::array_t<float> arr = random.attr("randn")(*shape_tuple).cast<py::array_t<float>>();
            float scale = (float)sqrt(param.shape[0]);
            arr = arr.attr("__truediv__")(scale).cast<py::array_t<float>>();
            return tf.attr("tensor")(arr);
        } else {
            py::object np = py::module::import("numpy");

            // Convert the shape vector to a tuple
            py::tuple shape_tuple = py::cast(param.shape);

            py::array_t<float> arr = np.attr("zeros")(shape_tuple).cast<py::array_t<float>>();
            return tf.attr("tensor")(arr);
        }
    }

    void initialize_parameters() {
        for (auto& module : _modules) {
            module.second.attr("initialize_parameters")();
        }

        for (auto& param : _parameters) {
            Parameter& p = py::cast<Parameter&>(param.second);
            py::object tensor = initialize_parameter(p);
            _parameters.attr("__setitem__")(param.first, tensor);
        }

        for (auto& array : _parameter_arrays) {
            ParameterArray& param_array = py::cast<ParameterArray&>(array.second);
            for (auto& param : param_array._parameters) {
                Parameter& p = py::cast<Parameter&>(param.second);
                py::object tensor = initialize_parameter(p);
                param_array._parameters.attr("__setitem__")(param.first, tensor);
            }
        }
    }

    py::list get_all_parameters() {
        py::list params;
        for (auto& module : _modules) {
            params += module.second.attr("get_all_parameters")();
        }
        for (auto& param : _parameters) {
            params.append(param.second);
        }
        for (auto& array : _parameter_arrays) {
            ParameterArray& param_array = py::cast<ParameterArray&>(array.second);
            for (auto& param : param_array._parameters) {
                params.append(param.second);
            }
        }
        return params;
    }

    py::list create_input(py::args args) {
        py::list inputs = get_all_parameters();
        inputs += args;
        return inputs;
    }

    void update_parameters(py::object parameter_values) {
        py::list params;
        if (py::isinstance<py::list>(parameter_values)) {
            params = parameter_values;
        } else if (py::isinstance<py::tuple>(parameter_values)) {
            params = py::list(parameter_values);
        } else {
            throw py::type_error("parameter_values must be a list or tuple");
        }

        int index = 0;

        std::function<void(Module&)> update_params;

        update_params = [&](Module& module) {
            for (auto& module_item : module._modules) {
                update_params(py::cast<Module&>(module_item.second));
            }

            for (auto& param : module._parameters) {
                if (index >= py::len(params)) {
                    throw py::index_error("Not enough values provided to update all parameters");
                }
                module._parameters[param.first] = params[index];
                index++;
            }

            for (auto& array : module._parameter_arrays) {
                ParameterArray& param_array = py::cast<ParameterArray&>(array.second);
                for (auto& param : param_array._parameters) {
                    if (index >= py::len(params)) {
                        throw py::index_error("Not enough values provided to update all parameters");
                    }
                    param_array._parameters[param.first] = params[index];
                    index++;
                }
            }
        };

        update_params(*this);
    }

    virtual py::object loss(py::object X, py::object Y) {
        throw std::runtime_error("Not implemented");
    }

    virtual py::object forward(py::object X) {
        throw std::runtime_error("Not implemented");
    }
};

}