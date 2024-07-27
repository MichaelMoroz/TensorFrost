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
    map<size_t, py::object> _parameters; //must be sorted

    py::object getitem(size_t index) {
        if (_parameters.contains(index)) {
            return _parameters[index];
        }
        throw py::index_error("Index " + std::to_string(index) + " is not in the ParameterArray");
    }

    void setitem(size_t index, py::object value) {
        _parameters[index] = value;
    }
};

class Module {
public:
    enum class AttributeType {
        None,
        Parameter,
        ParameterArray,
        Module
    };

    unordered_map<string, py::object> _attributes;
    unordered_map<string, AttributeType> _attribute_types;

    py::object tf;

    Module() {
        tf = py::module::import("TensorFrost");
    }

    py::object getattr(const std::string& name) {
        if (_attributes.contains(name)) {
            return _attributes[name];
        }
        throw py::attribute_error("TensorFrost Module object has no attribute with name '" + name + "'");
    }

    void setattr(const std::string& name, py::object value) {
        AttributeType type = AttributeType::None;
        if (py::isinstance<Parameter>(value)) {
            type = AttributeType::Parameter;
        } else if (py::isinstance<ParameterArray>(value)) {
            type = AttributeType::ParameterArray;
        } else if (py::isinstance<Module>(value)) {
            type = AttributeType::Module;
        }
        if(type == AttributeType::None && _attribute_types.contains(name)) {
            type = _attribute_types[name];
        }

        _attributes[name] = value;
        _attribute_types[name] = type;
    }

    bool hasattr(const std::string& name) {
        return _attributes.contains(name);
    }

    vector<pair<string, py::object>> get_attributes_of_type(AttributeType type) {
        vector<pair<string, py::object>> params;
        for (auto& attr : _attributes) {
            if (_attribute_types[attr.first] == type) {
                params.push_back(attr);
            }
        }
        return params;
    }

    virtual void assert_parameters() {}

    void initialize_input() {
        for (auto& module : get_attributes_of_type(AttributeType::Module)) {
            module.second.attr("initialize_input")();
        }

        for (auto& param : get_attributes_of_type(AttributeType::Parameter)) {
            Parameter& p = py::cast<Parameter&>(param.second);
            py::object tensor = tf.attr("input")(p.shape, p.dtype);
            setattr(param.first, tensor);
        }

        for (auto& array : get_attributes_of_type(AttributeType::ParameterArray)) {
            ParameterArray& param_array = py::cast<ParameterArray&>(array.second);
            for (auto& param : param_array._parameters) {
                Parameter& p = py::cast<Parameter&>(param.second);
                py::object tensor = tf.attr("input")(p.shape, p.dtype);
                param_array.setitem(param.first, tensor);
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
        for (auto& module : get_attributes_of_type(AttributeType::Module)) {
            module.second.attr("initialize_parameters")();
        }

        for (auto& param : get_attributes_of_type(AttributeType::Parameter)) {
            Parameter& p = py::cast<Parameter&>(param.second);
            py::object tensor = initialize_parameter(p);
            setattr(param.first, tensor);
        }

        for (auto& array : get_attributes_of_type(AttributeType::ParameterArray)) {
            ParameterArray& param_array = py::cast<ParameterArray&>(array.second);
            for (auto& param : param_array._parameters) {
                Parameter& p = py::cast<Parameter&>(param.second);
                py::object tensor = initialize_parameter(p);
                param_array.setitem(param.first, tensor);
            }
        }
    }

    py::list parameters() {
        py::list params;
        for (auto& module : get_attributes_of_type(AttributeType::Module)) {
            params += module.second.attr("parameters")();
        }

        for (auto& param : get_attributes_of_type(AttributeType::Parameter)) {
            params.append(param.second);
        }

        for (auto& array : get_attributes_of_type(AttributeType::ParameterArray)) {
            ParameterArray& param_array = py::cast<ParameterArray&>(array.second);
            for (auto& param : param_array._parameters) {
                params.append(param.second);
            }
        }
        return params;
    }

    py::list create_input(py::args args) {
        py::list inputs = parameters();
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
            for (auto& module_item : module.get_attributes_of_type(AttributeType::Module)) {
                update_params(py::cast<Module&>(module_item.second));
            }

            for (auto& param : module.get_attributes_of_type(AttributeType::Parameter)) {
                if (index >= py::len(params)) {
                    throw py::index_error("Not enough values provided to update all parameters");
                }
                module.setattr(param.first, params[index]);
                index++;
            }

            for (auto& array : module.get_attributes_of_type(AttributeType::ParameterArray)) {
                ParameterArray& param_array = py::cast<ParameterArray&>(array.second);
                for (auto& param : param_array._parameters) {
                    if (index >= py::len(params)) {
                        throw py::index_error("Not enough values provided to update all parameters");
                    }
                    param_array.setitem(param.first, params[index]);
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