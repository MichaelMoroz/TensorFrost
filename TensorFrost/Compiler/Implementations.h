#pragma once

#include "Backend/TensorMemory.h"
#include "Tensor/Tensor.h"

namespace TensorFrost {

const Tensor& ReduceGradientToShape(const Tensor& gradient, const Tensor& target);
int GetGradAxis(const Tensor& out, const Tensor& grad);

class NodeGrads
{
	unordered_map<ArgID, Tensor*, HashArgID> argument_gradients;
	unordered_map<ArgID, const Tensor*, HashArgID> argument_inputs;
public:
	//get element at index
	const Tensor& operator[](ArgID id) {
		return *argument_gradients[id];
	}

	bool Contains(ArgID id) {
		return argument_gradients.contains(id);
	}

	bool Contains(ArgType type, int index = 0) {
		return Contains(ArgID(type, index));
	}

	NodeGrads(Node* node, map<Node*, Tensor*> input_grads) {
		for(auto& [id, input] : node->args.inputs_) {
			argument_inputs[id] = input->GetTensor();
			if(input_grads.contains(input)) {
				argument_gradients[id] = input_grads[input];
			}
		}
	}

	void Add(ArgType type, int index, Tensor& tensor) {
		const Tensor* target = argument_inputs[ArgID(type, index)];
		Tensor& new_tensor = const_cast<Tensor&>(ReduceGradientToShape(tensor, *target));
		if(Contains(type, index)) {
			argument_gradients[ArgID(type, index)] = &(*argument_gradients[ArgID(type, index)] + new_tensor);
		} else {
			argument_gradients[ArgID(type, index)] = &new_tensor;
		}
	}

	Tensor* GetGrad(ArgID id) {
		if(Contains(id)) {
			return argument_gradients[id];
		} else {
			Tensor* zero_grad = &Tensor::Constant(argument_inputs[id]->GetShape(), 0.0f);
			argument_gradients[id] = zero_grad;
			return zero_grad;
		}
	}

	Tensor* GetGrad(ArgType type, int index) {
		return GetGrad(ArgID(type, index));
	}

	//add gradients to inputs
	template <typename... Args>
	void Add(Tensor& arg, Args&... args) {
		//by default these are ArgType::Input
		vector<Tensor*> inputs = vector<Tensor*>({ &arg, &args... });
		for (int i = 0; i < inputs.size(); i++) {
			Add(ArgType::Input, i, *inputs[i]);
		}
	}
};


typedef function<void(ArgumentManager&, Tensor&, Tensor&, NodeGrads&)> VJPGradientFunction;
typedef function<Tensors(map<int, const Tensor*> inputs, const Tensor* gradient, const Tensor* tensor)> AlgorithmVJPGradientFunction;

VJPGradientFunction GetVJPForOperation(string name);
void RegisterVJP(string name, VJPGradientFunction vjp);

//TODO JVPGradientFunction for forward mode autodiff

typedef function<void(Tensors& output, map<int, const Tensor*> inputs, const Tensor* tensor, vector<int> axes)> ImplementationFunction;

ImplementationFunction GetImplementationForOperation(string name);
void RegisterImplementation(string name, ImplementationFunction impl);

void RegisterAlgorithmicPrimitive(const string& name, vector<string> overloads,  ImplementationFunction impl, AlgorithmVJPGradientFunction vjp);

}
