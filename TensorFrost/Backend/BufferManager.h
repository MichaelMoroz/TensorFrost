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
#include <stdint.h>

namespace TensorFrost {
    using namespace std;

    extern "C" {
        struct TFBuffer {
            int size = 0;
        };
    }

    class BufferManager {
        TFBuffer* CreateBuffer(int size) {
            TFBuffer* buffer = new TFBuffer();
            buffer->size = size;
            return buffer;
        }

        TFBuffer* AllocateBuffer(int size) {
            TFBuffer* buffer = CreateBuffer(size);
            //add the buffer to the list of allocated buffers
            allocated_buffers[size].insert(buffer);
            return buffer;
        }

    public:
        const int MAX_UNUSED_TIME = 512;
        map<int, unordered_set<TFBuffer*>> allocated_buffers;
        map<TFBuffer*, int> unused_time;
        unordered_set<TFBuffer*> buffers_to_delete;
        unordered_set<TFBuffer*> used_buffers;

        BufferManager() {}
        void DeallocateBuffer(TFBuffer* buffer);
        void RemoveBuffer(TFBuffer* buffer);
        void UpdateTick();
        TFBuffer* TryAllocateBuffer(int size);
        uint32_t GetRequiredAllocatedStorage() const;
        uint32_t GetUnusedAllocatedStorage() const;
        ~BufferManager();
    };

}  // namespace TensorFrost
