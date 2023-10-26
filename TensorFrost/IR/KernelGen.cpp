#pragma once

#include "IR/KernelGen.h"

namespace TensorFrost {

bool IsBoundary(const Node* input, const Node* output, int arg_index,
                Argument::Type arg_type) {
    OpType input_type = input->op->GetOpType();
    OpType output_type = output->op->GetOpType();

	bool isFromScatter = input_type == OpType::Scatter;
	bool isToScatter = output_type == OpType::Scatter;
	bool isFromStore = input_type == OpType::Store;
	bool isFromOutput = input->is_output_; 

    if(isFromScatter || isFromStore || isFromOutput)
    {
        if(isFromScatter && isToScatter)
        {
            return false; //multiple scatters can be merged
        }
        else
        {
            return true;
        }
    }

    if (output_type == OpType::Load)
    {
        return arg_index == 0 && arg_type == Argument::Type::Input;
    }

    if (output_type == OpType::Scatter)
    {
	    return arg_index == 0 && arg_type == Argument::Type::Input;
	}

    //TODO write shape comparison function
    //if(input.tensor_.shape_ != output.tensor_.shape_)
    //{
    //    if(!isToScatter)
    //    {
    //        return true;
    //    }
    //}

    //input memory should not be inside work clusters
    if (input->name == "input_memory")
    {
        return true;
    }

    return false;
}

void IR::Clusterize() {
    int cluster_id = 0;
    Node* prev = nullptr;
    for(auto& node: nodes_)
    {
        // check if node is a cluster edge
        Tensor* tensor = node.tensor_;

        bool is_boundary = false;

        if(prev != nullptr)
        {
			if (prev->cluster_id_ == cluster_id && IsBoundary(prev, &node, -1, Argument::Type::None))
            {
                is_boundary = true;
            }
        }

        //go over all inputs
        for(auto& input: tensor->node->arguments_)
        {
            //check if input is the boundary of this cluster
			if (input.node_->cluster_id_ == cluster_id &&
			    IsBoundary(input.node_, &node, input.index_, input.type_))
            {
                is_boundary = true;
                break;
            }
        }

        if(is_boundary)
        {
            cluster_id++;
        }

        node.cluster_id_ = cluster_id;
        prev = &node;
    }
}

}   // namespace TensorFrost