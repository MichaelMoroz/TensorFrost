#pragma once

#include "Backend/TensorMemory.h"
#include "Tensor/Tensor.h"

namespace TensorFrost {

const Tensor& ReduceGradientToShape(const Tensor& gradient, const Tensor& target);
int GetGradAxis(const Tensor& out, const Tensor& grad);

class NodeGrads
{
	unordered_map<Node*, const Tensor*> stored_gradients;
	unordered_map<ArgID, Node*, HashArgID> arguments;
	unordered_map<ArgID, const Tensor*, HashArgID> argument_inputs;
public:
	//get element at index
	const Tensor& operator[](ArgID id) {
		return *stored_gradients[arguments[id]];
	}

	bool Contains(ArgID id) {
		return stored_gradients.contains(arguments[id]);
	}

	bool Contains(ArgType type, int index = 0) {
		return Contains(ArgID(type, index));
	}

	NodeGrads(Node* node, map<Node*, const Tensor*> input_grads) {
		try {
			for(auto& [id, input] : node->args.inputs_) {
				if (id.first == ArgType::Index || id.first == ArgType::Shape) {
					continue;
				}
				argument_inputs[id] = input->GetTensor();
				arguments[id] = input;
				if(input_grads.contains(input)) {
					stored_gradients[input] = &ReduceGradientToShape(*input_grads[input], *input->GetTensor());
				}
			}
		} catch (const std::exception& e) {
			throw std::runtime_error("Error in gradient initialization: " + string(e.what()));
		}
	}

	void Add(ArgType type, int index, const Tensor& tensor) {
		ArgID id = ArgID(type, index);
		const Tensor* target = argument_inputs[id];
		try {
			Tensor& new_tensor = const_cast<Tensor&>(ReduceGradientToShape(tensor, *target));
			if(Contains(type, index)) {
				auto& old_tensor = *stored_gradients[arguments[id]];
				old_tensor.StopFusion();
				auto& loaded = old_tensor; //TODO: temporary way to restrict fusion, remove after implementing split
				stored_gradients[arguments[id]] = &(loaded + new_tensor);
			} else {
				stored_gradients[arguments[id]] = &new_tensor;
			}
		} catch (const std::exception& e) {
			throw std::runtime_error("Error in gradient addition: " + string(e.what()));
		}
	}

	const Tensor* GetGrad(ArgID id) {
		if(Contains(id)) {
			return stored_gradients[arguments[id]];
		} else {
			IR* cur_ir = Tensor::GetEvaluationContext();
			const Tensor* input = argument_inputs[id];
			Tensor* zero_grad = nullptr;
			cur_ir->ExecuteExpressionAfter(input->node_, [&]() {
				zero_grad = &Tensor::Constant(argument_inputs[id]->GetShape(), 0.0f);
				stored_gradients[arguments[id]] = zero_grad;
			});
			return zero_grad;
		}
	}

	const Tensor* GetGrad(ArgType type, int index) {
		return GetGrad(ArgID(type, index));
	}

	//add gradients to inputs
	template <typename... Args>
	void Add(const Tensor& arg, Args&... args) {
		//by default these are ArgType::Input
		vector<const Tensor*> inputs = vector<const Tensor*>({ &arg, &args... });
		for (int i = 0; i < inputs.size(); i++) {
			Add(ArgType::Input, i, *inputs[i]);
		}
	}
};


typedef function<void(ArgumentManager&, const Tensor&, const Tensor&, NodeGrads&)> VJPGradientFunction;
typedef function<Tensors(map<int, const Tensor*> inputs, const Tensor* gradient, const Tensor* tensor)> AlgorithmVJPGradientFunction;

VJPGradientFunction GetVJPForOperation(string name);
void RegisterVJP(string name, VJPGradientFunction vjp);
bool HasDerivativeImplemented(string name);
//TODO JVPGradientFunction for forward mode autodiff

typedef function<void(Tensors& output, map<int, const Tensor*> inputs, const Tensor* tensor, vector<int> axes)> ImplementationFunction;

ImplementationFunction GetImplementationForOperation(string name);
void RegisterImplementation(string name, ImplementationFunction impl);

void RegisterAlgorithmicPrimitive(const string& name, vector<string> overloads,  ImplementationFunction impl, AlgorithmVJPGradientFunction vjp);

}
