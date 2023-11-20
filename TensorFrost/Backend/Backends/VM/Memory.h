#pragma once

#include <functional>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>

#include "../../TensorMemory.h"

namespace TensorFrost {

using namespace std;

class CPU_MemoryManager : public TensorMemoryManager
{
public:
    vector<uint> memory;

    TensorMemory* Allocate(const vector<int>& shape) override
    {
        int size = GetLinearSize(shape);
        Frame* frame = allocator.AllocateFrame(size);
        // reserve space in memory if needed
        if (frame->end > memory.size())
        {
			    memory.resize(frame->end);
        }

        TensorMemory* tensorMemory = new TensorMemory(shape, frame);
        allocated[frame] = tensorMemory;
        return tensorMemory;
    }

    TensorMemory* AllocateWithData(const vector<int>& shape, const vector<uint>& data) override
    {
        TensorMemory* tensorMemory = Allocate(shape);
        memcpy(memory.data() + tensorMemory->frame->start, data.data(), data.size() * sizeof(uint));
        return tensorMemory;
    }

    vector<uint> Readback(const TensorMemory* memory) override
    {
        vector<uint> data;
        data.resize(memory->GetSize());
        memcpy(data.data(), this->memory.data() + memory->frame->start, data.size() * sizeof(uint));
        return data;
    }

    void Free(TensorMemory* memory) override
    {
        Frame* frame = memory->frame;
        allocator.FreeFrame(*frame);
        allocated.erase(frame);
        delete memory;
    }

    ~CPU_MemoryManager() override
    {
        for (auto& pair : allocated)
        {
            delete pair.second;
        }
    }
};


}  // namespace TensorFrost