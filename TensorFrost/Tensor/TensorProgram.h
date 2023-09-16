#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>
#include <functional>
#include "Tensor.h"

namespace TensorFrost 
{

using namespace std;

class TensorProgram {
public:
    using EvaluateFunction = function<vector<Tensor>(vector<Tensor>)>;
    EvaluateFunction evaluate_callback;
    IR ir;
    bool debug = false;

    TensorProgram(EvaluateFunction evaluate) : evaluate_callback(evaluate) 
    {
        vector<Tensor> inputs = vector<Tensor>();
        ir = IR();
        CreateExecutionGraph(inputs);
    }

    void CreateExecutionGraph(vector<Tensor> inputs) 
    {
        //create new IR graph
        Tensor::SetIR(&ir);
        vector<Tensor> outputs = evaluate_callback(inputs);
    }
    
    vector<Tensor> Evaluate(vector<Tensor> inputs)
    {
		return vector<Tensor>();
    }
};

}