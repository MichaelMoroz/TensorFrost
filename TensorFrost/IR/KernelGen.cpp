#pragma once

#include "IR/KernelGen.h"

namespace TensorFrost {

//public static bool IsBoundary(TEdge edge)
//{
//    // Check if the 'From' node of the edge is a scatter or store function
//    bool isFromScatter = TensorOp.IsScatterOp(edge.From.type);
//    //check if the 'To' node is a scatter
//    bool isToScatter = TensorOp.IsScatterOp(edge.To.type);
//
//    bool isFromStore = TensorOp.IsStoreOp(edge.From.type);
//
//    bool isFromOutput = false;//edge.From.property.isOutput;
//
//    if(isFromScatter || isFromStore || isFromOutput)
//    {
//        if(isFromScatter && isToScatter)
//        {
//            return false; //multiple scatters can be merged
//        }
//        else
//        {
//            return true;
//        }
//    }
//
//    // Check if the 'To' node of the edge is a load function
//    bool isToLoad = TensorOp.IsLoadOp(edge.To.type);
//
//    //if to a load function, check edge is not the memory source argument (first argument)
//
//    if(isToLoad)
//    {
//        return edge.property == 0;
//    }
//    
//    //if its the memory source for a scatter then its also an edge
//    if(isToScatter)
//    {
//        return edge.property == 0;
//    }
//
//    //check if the sizes of the node tensors are different (if so, then its a boundary)
//    if(!Tensor.CompareShape(edge.From.property.shape.size, edge.To.property.shape.size))
//    {
//        //if not a scatter, then its a boundary
//        if(!isToScatter)
//        {
//            return true;
//        }
//    }
//
//    //also if any of the nodes is TensorOp.Type.memory then its a boundary
//    if(edge.From.type == TensorOp.Type.memory || edge.To.type == TensorOp.Type.memory)
//    {
//        return true;
//    }
//
//    return false;
//}

bool IsBoundary(const Tensor* input, Node& output)
{
    bool isFromScatter = input->op->GetOpType() == OpType::Scatter;
    bool isToScatter = output.tensor_->op->GetOpType() == OpType::Scatter;
    bool isFromStore = input->op->GetOpType() == OpType::Store;
	bool isFromOutput = false; //input->is_output_; //TODO move arguments to node

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

    bool isToLoad = output.tensor_->op->GetOpType() == OpType::Load;

    if(isToLoad)
    {
        return false;
    }

    if(isToScatter)
    {
        return false;
    }

    //TODO write shape comparison function
    //if(input.tensor_.shape_ != output.tensor_.shape_)
    //{
    //    if(!isToScatter)
    //    {
    //        return true;
    //    }
    //}

    if(input->name == "input_memory")
    {
        return true;
    }

    return false;
}

void IR::Clusterize() {
    int cluster_id = 0;
    for(auto& node: nodes_)
    {
        // check if node is a cluster edge
        Tensor* tensor = node.tensor_;

        //go over all inputs
        for(auto& input: tensor->inputs)
        {
			if (input.type == Argument::Type::Shape) continue;
			//if (input.type == Argument::Type::Index) continue;
            //check if input is a cluster edge
            if(IsBoundary(input.tensor, node))
            {
                //if so, increment cluster id
                cluster_id++;
                break;
            }
        }

        node.cluster_id_ = cluster_id;
    }
}

}   // namespace TensorFrost