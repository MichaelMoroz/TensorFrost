
#include <vector>

enum DataType
{
    f32,
    i32,
    u32,
    b1,
};

class Tensor 
{
private:

    template<typename... Args>
    static Tensor Op(const std::string& op, const Args*... args)
    {
        // convert the parameter pack to a std::vector
        std::vector<const Tensor*> tensors = { args... };
        
        //graph.AddOperation(op, tensors);
        return Tensor(tensors[0]->shape, 0.0);
    }

    //static TGraph graph;

public:
    // void ClearGraph() { graph.Clear(); }
    DataType type;
    std::vector<int> shape;

    // Constructor that takes a list of sizes for each dimension
    Tensor(std::vector<int> sizes, float value = 0.0)
    {
        shape = sizes;
        type = DataType::f32;
    }

    Tensor(float value)
    {
        shape = { 1 };
        type = DataType::f32;
    }

    Tensor(int value)
    {
        shape = { 1 };
        type = DataType::i32;
    }

    Tensor(float *data, std::vector<int> sizes)
    {
        shape = sizes;
        type = DataType::f32;
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

    Tensor operator+(const Tensor& other) const 
    {
        return Op("add", this, &other);
    }

	Tensor operator-(const Tensor& other) const
	{
		return Op("sub", this, &other);
	}

	Tensor operator*(const Tensor& other) const
	{
		return Op("mul", this, &other);
	}
    
	Tensor operator/(const Tensor& other) const
	{
		return Op("div", this, &other);
	}

	Tensor operator-() const
	{
		return Op("neg", this);
	}

    Tensor operator%(const Tensor& other) const
    {
        return Op("mod", this, &other);
    }

	Tensor operator>(const Tensor& other) const
	{
		return Op("gt", this, &other);
	}
    
	Tensor operator<(const Tensor& other) const
	{
		return Op("lt", this, &other);
	}
    
	Tensor operator>=(const Tensor& other) const
	{
		return Op("gte", this, &other);
	}
    
	Tensor operator<=(const Tensor& other) const
	{
		return Op("lte", this, &other);
	}
    
	Tensor operator==(const Tensor& other) const
	{
		return Op("eq", this, &other);
	}
    
    Tensor operator!=(const Tensor& other) const
    {
        return Op("neq", this, &other);
    }

    Tensor operator&&(const Tensor& other) const
    {
        return Op("and", this, &other);
    }

    Tensor operator||(const Tensor& other) const
    {
        return Op("or", this, &other);
    }

    Tensor operator!() const
    {
        return Op("not", this);
    }

    Tensor operator~() const
    {
        return Op("bnot", this);
    }

    Tensor operator&(const Tensor& other) const
    {
        return Op("band", this, &other);
    }

    Tensor operator|(const Tensor& other) const
    {
        return Op("bor", this, &other);
    }

    Tensor operator^(const Tensor& other) const
    {
        return Op("bxor", this, &other);
    }

    Tensor operator<<(const Tensor& other) const
    {
        return Op("blshift", this, &other);
    }

    Tensor operator>>(const Tensor& other) const
    {
        return Op("brshift", this, &other);
    }

    static Tensor ifcond(const Tensor& condition, const Tensor& ifTrue, const Tensor& ifFalse)
    {
        return Op("cond", &condition, &ifTrue, &ifFalse);
    }

    static Tensor constant(const std::vector<int>& shape, float value)
    {
        return Tensor(shape, value);
    }

    static Tensor sin(const Tensor& x)
    {
        return Op("sin", &x);
    }

    static Tensor cos(const Tensor& x)
    {
        return Op("cos", &x);
    }

    static Tensor tan(const Tensor& x)
    {
        return Op("tan", &x);
    }

    static Tensor asin(const Tensor& x)
    {
        return Op("asin", &x);
    }

    static Tensor acos(const Tensor& x)
    {
        return Op("acos", &x);
    }

    static Tensor atan(const Tensor& x)
    {
        return Op("atan", &x);
    }

    static Tensor sinh(const Tensor& x)
    {
        return Op("sinh", &x);
    }

    static Tensor cosh(const Tensor& x)
    {
        return Op("cosh", &x);
    }

    static Tensor tanh(const Tensor& x)
    {
        return Op("tanh", &x);
    }

    static Tensor asinh(const Tensor& x)
    {
        return Op("asinh", &x);
    }

    static Tensor acosh(const Tensor& x)
    {
        return Op("acosh", &x);
    }

    static Tensor atanh(const Tensor& x)
    {
        return Op("atanh", &x);
    }

    static Tensor exp(const Tensor& x)
    {
        return Op("exp", &x);
    }

    static Tensor log(const Tensor& x)
    {
        return Op("log", &x);
    }

    static Tensor log2(const Tensor& x)
    {
        return Op("log2", &x);
    }

    static Tensor exp2(const Tensor& x)
    {
        return Op("exp2", &x);
    }

    static Tensor sqrt(const Tensor& x)
    {
        return Op("sqrt", &x);
    }

    static Tensor sqr(const Tensor& x)
    {
        return Op("sqr", &x);
    }

    static Tensor rsqrt(const Tensor& x)
    {
        return Op("rsqrt", &x);
    }

    static Tensor rcp(const Tensor& x)
    {
        return Op("rcp", &x);
    }

    static Tensor abs(const Tensor& x)
    {
        return Op("abs", &x);
    }

    static Tensor sign(const Tensor& x)
    {
        return Op("sign", &x);
    }

    static Tensor floor(const Tensor& x)
    {
        return Op("floor", &x);
    }

    static Tensor ceil(const Tensor& x)
    {
        return Op("ceil", &x);
    }

    static Tensor round(const Tensor& x)
    {
        return Op("round", &x);
    }

    static Tensor trunc(const Tensor& x)
    {
        return Op("trunc", &x);
    }

    static Tensor frac(const Tensor& x)
    {
        return Op("frac", &x);
    }

    static Tensor clamp(const Tensor& x, const Tensor& min, const Tensor& max)
    {
        return Op("clamp", &x, &min, &max);
    }

    static Tensor pow(const Tensor& x, const Tensor& y)
    {
        return Op("pow", &x, &y);
    }

    static Tensor min(const Tensor& x, const Tensor& y)
    {
        return Op("min", &x, &y);
    }

    static Tensor max(const Tensor& x, const Tensor& y)
    {
        return Op("max", &x, &y);
    }

    static Tensor mod(const Tensor& x, const Tensor& y)
    {
        return Op("mod", &x, &y);
    }

    static Tensor atan2(const Tensor& x, const Tensor& y)
    {
        return Op("atan2", &x, &y);
    }

    static Tensor lerp(const Tensor& x, const Tensor& y, const Tensor& a)
    {
        return Op("lerp", &x, &y, &a);
    }

    static Tensor fma(const Tensor& x, const Tensor& y, const Tensor& z)
    {
        return Op("fma", &x, &y, &z);
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