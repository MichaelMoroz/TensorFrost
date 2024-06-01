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
            size_t size = 0;
            bool up_to_date = true;
            bool read_only = false;
            //add type descriptor (for special kinds of buffers)
        };
    }

    class BufferManager {
        TFBuffer* CreateBuffer(size_t size) {
            TFBuffer* buffer = new TFBuffer();
            buffer->size = size;
            return buffer;
        }

        TFBuffer* AllocateBuffer(size_t size) {
            TFBuffer* buffer = CreateBuffer(size);
            //add the buffer to the list of allocated buffers
            allocated_buffers[size].insert(buffer);
            return buffer;
        }

    public:
        const int MAX_UNUSED_TIME = 512;
        map<size_t, unordered_set<TFBuffer*>> allocated_buffers;
        map<TFBuffer*, int> unused_time;
        unordered_set<TFBuffer*> buffers_to_delete;
        unordered_set<TFBuffer*> used_buffers;

        BufferManager() {}
        void DeallocateBuffer(TFBuffer* buffer);
        void RemoveBuffer(TFBuffer* buffer);
        void UpdateTick();
        TFBuffer* TryAllocateBuffer(size_t size);
        size_t GetRequiredAllocatedStorage() const;
        size_t GetUnusedAllocatedStorage() const;
        ~BufferManager();
    };

}  // namespace TensorFrost
