#include <TensorFrost.h>
#include <Frontend/Python/PyTensor.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace TensorFrost {

namespace py = pybind11;

class Parameter {
public:
    std::vector<int> shape;
    TFDataFormat dtype;
    float random_scale;
    float random_offset;
    string debug_name;
    bool optimize;

    Parameter(const std::vector<int>& shape, TFDataFormat dtype, float random_scale = -1.0f, float random_offset = 0.0f, bool requires_grad = true)
        : shape(shape), dtype(dtype), random_scale(random_scale), random_offset(random_offset), optimize(requires_grad), debug_name("") {}

    bool CanBeInitialized() {
        for (int i = 0; i < shape.size(); i++) {
            if (shape[i] == -1) {
                return false;
            }
        }
        return true;
    }
};

class ParameterArray {
public:
    map<size_t, py::object> _parameters; //must be sorted
    map<size_t, bool> _requires_grad;

    py::object getitem(size_t index) {
        if (_parameters.contains(index)) {
            return _parameters[index];
        }
        throw py::index_error("Index " + std::to_string(index) + " is not in the ParameterArray");
    }

    void setitem(size_t index, py::object value) {
        _parameters[index] = value;
        if (py::isinstance<Parameter>(value)) {
            _requires_grad[index] = py::cast<Parameter&>(value).optimize;
        }
    }

    vector<pair<size_t, py::object>> items() {
        vector<pair<size_t, py::object>> params;
        for (auto& param : _parameters) {
            params.push_back({param.first, param.second});
        }
        return params;
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

    map<string, py::object> _attributes;
    map<string, AttributeType> _attribute_types;
    map<string, bool> _optimize;
    vector<string> _attribute_order;
    bool optimize = true;

    py::object tf;

    Module(bool requires_grad = true) : optimize(requires_grad) {
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
        bool optimize = true;
        if (py::isinstance<Parameter>(value)) {
            type = AttributeType::Parameter;
            optimize = py::cast<Parameter&>(value).optimize;
            //if not float then can't optimize
            if(py::cast<Parameter&>(value).dtype.type != TFType::Float) {
                optimize = false;
            }
        } else if (py::isinstance<ParameterArray>(value)) {
            type = AttributeType::ParameterArray;
        } else if (py::isinstance<Module>(value)) {
            type = AttributeType::Module;
            optimize = py::cast<Module&>(value).optimize;
        } else if (py::isinstance<PyTensor>(value)) {
            PyTensor& tensor = py::cast<PyTensor&>(value);
            tensor.Get().node_->debug_name = name;
        }

        bool already_exists = _attributes.contains(name);
        if(type == AttributeType::None && already_exists) {
            type = _attribute_types[name];
            optimize = _optimize[name];
        }

        _attributes[name] = value;
        _attribute_types[name] = type;
        _optimize[name] = optimize && this->optimize;
        if (!already_exists) _attribute_order.push_back(name);
    }

    bool hasattr(const std::string& name) {
        return _attributes.contains(name);
    }

    bool param_requires_grad(const std::string& name) {
        if (_optimize.contains(name)) {
            return _optimize[name];
        }
        return optimize;
    }

    vector<pair<string, py::object>> get_attributes_of_type(AttributeType type) {
        vector<pair<string, py::object>> params;
        for (auto& attr : _attribute_order) {
            if (_attribute_types[attr] == type) {
                params.push_back({attr, _attributes[attr]});
            }
        }
        return params;
    }

    virtual void assert_parameters() {}

    py::object initialize_parameter_native(Parameter& param) {
        //Convert to list
        vector<py::object> shape_list;
        for (int i = 0; i < param.shape.size(); i++) {
            shape_list.push_back(tf.attr("const")(param.shape[i]));
        }
        py::object result;
        if(param.dtype.type == TFType::Float) {
            float shape_sum = 0.0f;
            for (int i = 0; i < param.shape.size(); i++) {
                shape_sum += (float)param.shape[i];
            }
            float scale = sqrt(6.0f / shape_sum);
            if(param.random_scale >= 0.0f) {
                scale = param.random_scale;
            }
            result = tf.attr("random_value")(shape_list, tf.attr("const")(0u));
            result = result.attr("__mul__")(py::float_(2.0f));
            result = result.attr("__sub__")(py::float_(1.0f));
            result = result.attr("__mul__")(py::float_(scale));
            result = result.attr("__add__")(py::float_(param.random_offset));
        } else if (param.dtype.type == TFType::Int) { //just use zeros
            result = tf.attr("const")(0, shape_list);
        } else if (param.dtype.type == TFType::Uint) { //just use zeros
            result = tf.attr("const")(0, shape_list);
        } else { //just use zeros
            result = tf.attr("const")(false, shape_list);
        }
        if(param.debug_name != "") {
            result = result.attr("set_debug_name")(param.debug_name);
        }
        return result;
    }

    void initialize_parameters_native() {
        for (auto& module : get_attributes_of_type(AttributeType::Module)) {
            module.second.attr("initialize_parameters_native")();
        }

        for (auto& param : get_attributes_of_type(AttributeType::Parameter)) {
            if(!py::isinstance<Parameter>(param.second)) {
                continue;
            }

            Parameter& p = py::cast<Parameter&>(param.second);
            if (!p.CanBeInitialized()) {
                //replace all -1 shapes with 1
                for (int i = 0; i < p.shape.size(); i++) {
                    if (p.shape[i] == -1) {
                        p.shape[i] = 1;
                    }
                }
            }
            py::object tensor = initialize_parameter_native(p);
            setattr(param.first, tensor);
        }

        for (auto& array : get_attributes_of_type(AttributeType::ParameterArray)) {
            ParameterArray& param_array = py::cast<ParameterArray&>(array.second);
            for (auto& param : param_array._parameters) {
                if(!py::isinstance<Parameter>(param.second)) {
                    continue;
                }
                Parameter& p = py::cast<Parameter&>(param.second);
                if (!p.CanBeInitialized()) {
                    continue;
                }
                py::object tensor = initialize_parameter_native(p);
                param_array.setitem(param.first, tensor);
            }
        }
    }

    void initialize_input() {
        for (auto& module : get_attributes_of_type(AttributeType::Module)) {
            module.second.attr("initialize_input")();
        }

        for (auto& param : get_attributes_of_type(AttributeType::Parameter)) {
            Parameter& p = py::cast<Parameter&>(param.second);
            py::object tensor = tf.attr("input")(p.shape, p.dtype);
            if(p.debug_name != "") {
                tensor = tensor.attr("set_debug_name")(p.debug_name);
            }
            setattr(param.first, tensor);
        }

        for (auto& array : get_attributes_of_type(AttributeType::ParameterArray)) {
            ParameterArray& param_array = py::cast<ParameterArray&>(array.second);
            for (auto& param : param_array._parameters) {
                Parameter& p = py::cast<Parameter&>(param.second);
                py::object tensor = tf.attr("input")(p.shape, p.dtype);
                if(p.debug_name != "") {
                    tensor = tensor.attr("set_debug_name")(p.debug_name);
                }
                param_array.setitem(param.first, tensor);
            }
        }

        assert_parameters();
    }

    py::object initialize_parameter(Parameter& param) {
        py::object np = py::module::import("numpy");
        py::object random = np.attr("random");

        // Convert the shape vector to a tuple
        py::tuple shape_tuple = py::cast(param.shape);

        if(param.dtype.type == TFType::Float) {
            // Generate uniform random values instead of normal
            py::array_t<float> arr = random.attr("uniform")(-1.0f, 1.0f, shape_tuple).cast<py::array_t<float>>();
            float shape_sum = 0.0f;
            for (int i = 0; i < param.shape.size(); i++) {
                shape_sum += (float)param.shape[i];
            }
            float scale = sqrt(6.0f / shape_sum);
            if(param.random_scale >= 0.0f) {
                scale = param.random_scale;
            }

            arr = arr.attr("__mul__")(py::float_(scale));
            arr = arr.attr("__add__")(py::float_(param.random_offset));
            return tf.attr("tensor")(arr);
        } else if (param.dtype.type == TFType::Int) { //just use zeros
            py::array_t<int> arr = np.attr("zeros")(shape_tuple).cast<py::array_t<int>>();
            return tf.attr("tensor")(arr);
        } else if (param.dtype.type == TFType::Uint) { //just use zeros
            py::array_t<unsigned int> arr = np.attr("zeros")(shape_tuple).cast<py::array_t<unsigned int>>();
            return tf.attr("tensor")(arr);
        } else { //just use zeros
            py::array_t<bool> arr = np.attr("zeros")(shape_tuple).cast<py::array_t<bool>>();
            return tf.attr("tensor")(arr);
        }
    }

    void initialize_parameters() {
        for (auto& module : get_attributes_of_type(AttributeType::Module)) {
            module.second.attr("initialize_parameters")();
        }

        for (auto& param : get_attributes_of_type(AttributeType::Parameter)) {
            if(!py::isinstance<Parameter>(param.second)) {
                continue;
            }

            Parameter& p = py::cast<Parameter&>(param.second);
            if (!p.CanBeInitialized()) {
                continue;
            }
            py::object tensor = initialize_parameter(p);
            setattr(param.first, tensor);
        }

        for (auto& array : get_attributes_of_type(AttributeType::ParameterArray)) {
            ParameterArray& param_array = py::cast<ParameterArray&>(array.second);
            for (auto& param : param_array._parameters) {
                if(!py::isinstance<Parameter>(param.second)) {
                    continue;
                }
                Parameter& p = py::cast<Parameter&>(param.second);
                if (!p.CanBeInitialized()) {
                    continue;
                }
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

    py::list named_parameters() {
        py::list params;
        for (auto& module : get_attributes_of_type(AttributeType::Module)) {
            params += module.second.attr("named_parameters")();
        }

        for (auto& param : get_attributes_of_type(AttributeType::Parameter)) {
            params.append(py::make_tuple(param.first, param.second));
        }

        for (auto& array : get_attributes_of_type(AttributeType::ParameterArray)) {
            ParameterArray& param_array = py::cast<ParameterArray&>(array.second);
            for (auto& param : param_array._parameters) {
                params.append(py::make_tuple(array.first + "[" + std::to_string(param.first) + "]", param.second));
            }
        }
        return params;
    }

    py::list requires_grads_list() {
        py::list requires_grads;
        for (auto& module : get_attributes_of_type(AttributeType::Module)) {
            requires_grads.append( param_requires_grad(module.first) );
        }

        for (auto& param : get_attributes_of_type(AttributeType::Parameter)) {
            requires_grads.append( param_requires_grad(param.first) );
        }

        for (auto& array : get_attributes_of_type(AttributeType::ParameterArray)) {
            ParameterArray& param_array = py::cast<ParameterArray&>(array.second);
            bool requires_grad = param_requires_grad(array.first);
            for (auto& param : param_array._parameters) {
                requires_grads.append( param_array._requires_grad[param.first] && requires_grad );
            }
        }
        return requires_grads;
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
                    throw py::index_error("Provided more than " + std::to_string(index) + " values, but expected " + std::to_string(py::len(params)));
                }
                module.setattr(param.first, params[index]);
                index++;
            }

            for (auto& array : module.get_attributes_of_type(AttributeType::ParameterArray)) {
                ParameterArray& param_array = py::cast<ParameterArray&>(array.second);
                for (auto& param : param_array._parameters) {
                    if (index >= py::len(params)) {
                        throw py::index_error("Provided more than " + std::to_string(index) + " values, but expected " + std::to_string(py::len(params)));
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