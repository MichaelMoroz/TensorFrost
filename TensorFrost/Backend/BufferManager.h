#pragma once

#ifdef linux
#include <cstdint>
#endif

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace TensorFrost {
    using namespace std;

    extern "C" {
        struct Buffer {
            int size = 0;
        };
    }

    class BufferManager {
        Buffer* CreateBuffer(int size) {
            Buffer* buffer = new Buffer();
            buffer->size = size;
            return buffer;
        }

        Buffer* AllocateBuffer(int size) {
            Buffer* buffer = CreateBuffer(size);
            //add the buffer to the list of allocated buffers
            allocated_buffers[size].insert(buffer);
            return buffer;
        }

    public:
        map<int, unordered_set<Buffer*>> allocated_buffers;
        unordered_set<Buffer*> used_buffers;

        BufferManager() {}
        void DeallocateBuffer(Buffer* buffer);
        void RemoveBuffer(Buffer* buffer);
        Buffer* TryAllocateBuffer(int size);
        uint32_t GetRequiredAllocatedStorage() const;
        uint32_t GetUnusedAllocatedStorage() const;
        ~BufferManager();
    };

}  // namespace TensorFrost
