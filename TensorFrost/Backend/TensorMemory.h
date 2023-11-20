#pragma once

#include <functional>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>

#include "../Tensor/Tensor.h"
#include "FrameAllocator.h"

namespace TensorFrost {

using namespace std;

int GetLinearSize(const vector<int>& shape);

class TensorMemory
{
public:
    vector<int> shape;
    Frame* frame;
    
    TensorMemory(const vector<int>& shape, Frame* frame)
        : shape(shape), frame(frame)
    {
    }

    int GetSize() const
    {
        return GetLinearSize(shape);
    }

    vector<int> GetShape() const
	{
		return shape;
	}
};

class TensorMemoryManager
{
public:
    FrameAllocator allocator;
    map<Frame*, TensorMemory*> allocated;

    virtual TensorMemory* Allocate(const vector<int>& shape) = 0;
    virtual TensorMemory* AllocateWithData(const vector<int>& shape, const vector<uint>& data) = 0;
    virtual vector<uint> Readback(const TensorMemory* memory) = 0;
    virtual void Free(TensorMemory* memory) = 0;

    virtual ~TensorMemoryManager() = default;
};

}  // namespace TensorFrost