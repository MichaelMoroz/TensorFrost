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

class TensorMemory;

class TensorMemoryManager {
 public:
	FrameAllocator allocator;
	map<Frame*, TensorMemory*> allocated;

	virtual TensorMemory* Allocate(const vector<int>& shape) = 0;
	virtual TensorMemory* AllocateWithData(const vector<int>& shape,
	                                       const vector<uint>& data) = 0;
	virtual vector<uint> Readback(const TensorMemory* memory) = 0;
	virtual void Free(TensorMemory* memory) = 0;

    uint32_t GetAllocatedSize() const
	{   
        return allocator.GetRequiredAllocatedStorage();
	}

	virtual ~TensorMemoryManager() = default;
};

class TensorMemory
{
public:
    vector<int> shape;
    TensorMemoryManager* manager;
    Frame* frame;
    
    TensorMemory(const vector<int>& shape, Frame* frame,
	               TensorMemoryManager* used_manager)
	      : shape(shape), frame(frame), manager(used_manager)
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

    ~TensorMemory()
    { 
        manager->Free(this);
    }
};



}  // namespace TensorFrost