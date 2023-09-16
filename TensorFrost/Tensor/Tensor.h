#pragma once

#include <..\..\TensorFrost\Utility\Utility.h>
#include <..\..\TensorFrost\IR\IR.h>
#include <vector>
#include <list>
#include <unordered_map>

namespace TensorFrost
{
class Tensor;

enum DimensionType
{
    // Tensor dimensions
    T_Thread,
    T_Loop,

    // Compute dimensions (for parallelism)
    C_OuterLoop, // outermost loop
    C_Block, // workgroups
    C_Group, // workitems
    C_InnerLoop,  // innermost loop (per workitem)
};

class Dimension
{
public:
    int size = 1;
    int min_size = -1; // -1 means no minimum
    int max_size = -1; // -1 means no maximum

    DimensionType type = DimensionType::T_Thread;
};

class Shape
{
public:
    std::vector<Dimension> dimensions;

    Shape(std::vector<Dimension> dimensions)
    {
        this->dimensions = dimensions;
    }

    template<typename... Args>
    Shape(int size, const Args&... args)
    {
        dimensions.push_back(Dimension());
        dimensions[0].size = size;
        AddDimensions(args...);
    }

    int GetSize() const
    {
        int size = 1;
        for (int i = 0; i < dimensions.size(); i++)
        {
            size *= dimensions[i].size;
        }
        return size;
    }

    std::vector<int> GetShape()
    {
        std::vector<int> shape = std::vector<int>();
        for (int i = 0; i < dimensions.size(); i++)
        {
            shape.push_back(dimensions[i].size);
        }
        return shape;
    }
};

class Argument 
{
public:
    enum Type
    {
        Input,
        RefCopy,
        Loop,
    };

    Type type;
    Tensor* tensor;
    int index;

    Argument(Type type, Tensor* tensor, int index)
    {
        this->type = type;
        this->tensor = tensor;
        this->index = index;
    }
};

class Tensor
{
private:
    template<typename... Args>
    static shared_ptr<Tensor> Op(const std::string& op, const Args*... args)
    {
        // convert the parameter pack to a std::vector
        std::vector<const Tensor*> tensors = { args... };

        // create argument list
        std::vector<Argument> arguments = std::vector<Argument>();

        for (int i = 0; i < tensors.size(); i++)
        {
            arguments.push_back(Argument(Argument::Type::Input, (Tensor*)tensors[i], i));
        }

        // get the shape of the output tensor
        std::vector<int> shape = tensors[0]->shape;

        shared_ptr<Tensor> output = make_shared<Tensor>(shape, arguments, op);

        // create the output tensor
        AddToGraph(output);
        return output;
    }

    static shared_ptr<Tensor> Static(const string& op, vector<int> shape)
    {
        shared_ptr<Tensor> output = make_shared<Tensor>(shape, vector<Argument>(), op);
        AddToGraph(output);
        return output;
    }

    static IR* graph;

    static void AddToGraph(shared_ptr<Tensor> node)
    {
        //check if IR is not null
        if (graph == nullptr)
        {
            throw std::runtime_error("Operation cannot be added to graph because the graph is null");
        }

        graph->AddNode(node);
    }

public:
    static void SetIR(IR* ir)
	{
		graph = ir;
	}

    string GetConstantString()
    {
        if (name == "const")
        {
            switch (type)
            {
            case DataType::f32:
                return to_string(AsFloat(data[0]));
            case DataType::i32:
                return to_string(AsInt(data[0]));
            case DataType::u32:
                return to_string(data[0]);
            }
        }
        else
        {
            return "";
        }
	}

    string name = "";
    DataType type = DataType::f32;
    std::vector<Argument> inputs;
    std::vector<int> shape;
    std::vector<uint> data;

    // Main constructor
    Tensor(vector<int> shape, vector<Argument> inputs, string name = "", DataType type = DataType::f32)
    {
        this->shape = shape;
        this->inputs = inputs;
        this->name = name;
        this->type = type;
    }


    // Constructor that takes a list of sizes for each dimension
    Tensor(std::vector<int> sizes, float value = 0.0) : Tensor(sizes, vector<Argument>(), "const", DataType::f32)
    {
        data = std::vector<uint>(1, AsUint(value));
    }

    Tensor(std::vector<int> sizes, int value) : Tensor(sizes, vector<Argument>(), "const", DataType::i32)
    {
        data = std::vector<uint>(1, AsUint(value));
    }

    Tensor(std::vector<int> sizes, uint value) : Tensor(sizes, vector<Argument>(), "const", DataType::u32)
    {
        data = std::vector<uint>(1, value);
    }

    Tensor(float value) : Tensor({ 1 }, value)
    {
    }

    Tensor(int value) : Tensor({ 1 }, value)
    {
    }

    Tensor(uint value) : Tensor({ 1 }, value)
    {
    }

    Tensor(float* data, std::vector<int> sizes) : Tensor(sizes, vector<Argument>(), "const_memory", DataType::f32)
    {
        uint data_count = Size();
        for (int i = 0; i < data_count; i++)
        {
            this->data.push_back(AsUint(data[i]));
        }
    }

    // tensor factory methods
    static Tensor& Constant(std::vector<int> shape, float value)
    {
        shared_ptr<Tensor> output = Static("const", shape);
        output->data = std::vector<uint>(1, AsUint(value));
        output->type = DataType::f32;
        return *output;
    }

    static Tensor& Constant(std::vector<int> shape, int value)
    {
        shared_ptr<Tensor> output = Static("const", shape);
        output->data = std::vector<uint>(1, AsUint(value));
        output->type = DataType::i32;
        return *output;
    }

    static Tensor& Constant(std::vector<int> shape, uint value)
    {
        shared_ptr<Tensor> output = Static("const", shape);
        output->data = std::vector<uint>(1, value);
        output->type = DataType::u32;
        return *output;
    }

    static Tensor& Constant(std::vector<int> shape, float *data)
    {
        shared_ptr<Tensor> output = Static("const_memory", shape);
        uint data_count = Size(shape);
        for (int i = 0; i < data_count; i++)
        {
            output->data.push_back(AsUint(data[i]));
        }
        return *output;
    }

    //destructor
	~Tensor()
	{
		
	}

    static int Size(const Tensor& tensor)
    {
        int size = 1;
        for (int i = 0; i < tensor.shape.size(); i++)
        {
            size *= tensor.shape[i];
        }
        return size;
    }

    int Size() const
    {
        return Size(*this);
    }

    // Method to get a value at a specific index
    double get(const std::vector<int>& indices) const 
    {
        return 0.0;
    }

    // Method to set a value at a specific index
    void set(const std::vector<int>& indices, double value) 
    {
        
    }

    // Overload for operator[]
    double operator[](const std::vector<int>& indices) const 
    {
        return get(indices);
    }

    Tensor& operator+(const Tensor& other) const 
    {
        return *Op("add", this, &other);
    }

	Tensor& operator-(const Tensor& other) const
	{
		return *Op("sub", this, &other);
	}

	Tensor& operator*(const Tensor& other) const
	{
		return *Op("mul", this, &other);
	}
    
	Tensor& operator/(const Tensor& other) const
	{
		return *Op("div", this, &other);
	}

	Tensor& operator-() const
	{
		return *Op("neg", this);
	}

    Tensor& operator%(const Tensor& other) const
    {
        return *Op("mod", this, &other);
    }

	Tensor& operator>(const Tensor& other) const
	{
		return *Op("gt", this, &other);
	}
    
	Tensor& operator<(const Tensor& other) const
	{
		return *Op("lt", this, &other);
	}
    
	Tensor& operator>=(const Tensor& other) const
	{
		return *Op("gte", this, &other);
	}
    
	Tensor& operator<=(const Tensor& other) const
	{
		return *Op("lte", this, &other);
	}
    
	Tensor& operator==(const Tensor& other) const
	{
		return *Op("eq", this, &other);
	}
    
    Tensor& operator!=(const Tensor& other) const
    {
        return *Op("neq", this, &other);
    }

    Tensor& operator&&(const Tensor& other) const
    {
        return *Op("and", this, &other);
    }

    Tensor& operator||(const Tensor& other) const
    {
        return *Op("or", this, &other);
    }

    Tensor& operator!() const
    {
        return *Op("not", this);
    }

    Tensor& operator~() const
    {
        return *Op("bnot", this);
    }

    Tensor& operator&(const Tensor& other) const
    {
        return *Op("band", this, &other);
    }

    Tensor& operator|(const Tensor& other) const
    {
        return *Op("bor", this, &other);
    }

    Tensor& operator^(const Tensor& other) const
    {
        return *Op("bxor", this, &other);
    }

    Tensor& operator<<(const Tensor& other) const
    {
        return *Op("blshift", this, &other);
    }

    Tensor& operator>>(const Tensor& other) const
    {
        return *Op("brshift", this, &other);
    }

    static Tensor& ifcond(const Tensor& condition, const Tensor& ifTrue, const Tensor& ifFalse)
    {
        return *Op("cond", &condition, &ifTrue, &ifFalse);
    }

    static Tensor& constant(const std::vector<int>& shape, float value)
    {
        return *new Tensor(shape, value);
    }

    static Tensor& sin(const Tensor& x)
    {
        return *Op("sin", &x);
    }

    static Tensor& cos(const Tensor& x)
    {
        return *Op("cos", &x);
    }

    static Tensor& tan(const Tensor& x)
    {
        return *Op("tan", &x);
    }

    static Tensor& asin(const Tensor& x)
    {
        return *Op("asin", &x);
    }

    static Tensor& acos(const Tensor& x)
    {
        return *Op("acos", &x);
    }

    static Tensor& atan(const Tensor& x)
    {
        return *Op("atan", &x);
    }

    static Tensor& sinh(const Tensor& x)
    {
        return *Op("sinh", &x);
    }

    static Tensor& cosh(const Tensor& x)
    {
        return *Op("cosh", &x);
    }

    static Tensor& tanh(const Tensor& x)
    {
        return *Op("tanh", &x);
    }

    static Tensor& asinh(const Tensor& x)
    {
        return *Op("asinh", &x);
    }

    static Tensor& acosh(const Tensor& x)
    {
        return *Op("acosh", &x);
    }

    static Tensor& atanh(const Tensor& x)
    {
        return *Op("atanh", &x);
    }

    static Tensor& exp(const Tensor& x)
    {
        return *Op("exp", &x);
    }

    static Tensor& log(const Tensor& x)
    {
        return *Op("log", &x);
    }

    static Tensor& log2(const Tensor& x)
    {
        return *Op("log2", &x);
    }

    static Tensor& exp2(const Tensor& x)
    {
        return *Op("exp2", &x);
    }

    static Tensor& sqrt(const Tensor& x)
    {
        return *Op("sqrt", &x);
    }

    static Tensor& sqr(const Tensor& x)
    {
        return *Op("sqr", &x);
    }

    static Tensor& rsqrt(const Tensor& x)
    {
        return *Op("rsqrt", &x);
    }

    static Tensor& rcp(const Tensor& x)
    {
        return *Op("rcp", &x);
    }

    static Tensor& abs(const Tensor& x)
    {
        return *Op("abs", &x);
    }

    static Tensor& sign(const Tensor& x)
    {
        return *Op("sign", &x);
    }

    static Tensor& floor(const Tensor& x)
    {
        return *Op("floor", &x);
    }

    static Tensor& ceil(const Tensor& x)
    {
        return *Op("ceil", &x);
    }

    static Tensor& round(const Tensor& x)
    {
        return *Op("round", &x);
    }

    static Tensor& trunc(const Tensor& x)
    {
        return *Op("trunc", &x);
    }

    static Tensor& frac(const Tensor& x)
    {
        return *Op("frac", &x);
    }

    static Tensor& clamp(const Tensor& x, const Tensor& min, const Tensor& max)
    {
        return *Op("clamp", &x, &min, &max);
    }

    static Tensor& pow(const Tensor& x, const Tensor& y)
    {
        return *Op("pow", &x, &y);
    }

    static Tensor& min(const Tensor& x, const Tensor& y)
    {
        return *Op("min", &x, &y);
    }

    static Tensor& max(const Tensor& x, const Tensor& y)
    {
        return *Op("max", &x, &y);
    }

    static Tensor& mod(const Tensor& x, const Tensor& y)
    {
        return *Op("mod", &x, &y);
    }

    static Tensor& atan2(const Tensor& x, const Tensor& y)
    {
        return *Op("atan2", &x, &y);
    }

    static Tensor& lerp(const Tensor& x, const Tensor& y, const Tensor& a)
    {
        return *Op("lerp", &x, &y, &a);
    }

    static Tensor& fma(const Tensor& x, const Tensor& y, const Tensor& z)
    {
        return *Op("fma", &x, &y, &z);
    }
};

class IndexedTensor 
{
public:
    Tensor* value;
    std::vector<Tensor*> indices;

    IndexedTensor(Tensor* value, std::vector<Tensor*> indices)
    {
        this->value = value;
        this->indices = indices;
    }
};

}
