#pragma once

#ifdef linux
#include <cstdint>
#endif

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
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
        unordered_map<int, unordered_set<Buffer*>> allocated_buffers;
        unordered_set<Buffer*> used_buffers;

        BufferManager() {}

        void DeallocateBuffer(Buffer* buffer) {
            used_buffers.erase(buffer);
        }

        void RemoveBuffer(Buffer* buffer) {
            for(auto& [size, buffers]: allocated_buffers) {
                buffers.erase(buffer);
            }
            DeallocateBuffer(buffer);
            delete buffer;
        }

        Buffer* TryAllocateBuffer(int size) {
            //try to find a non-used buffer of the correct size
            Buffer* buffer = nullptr;
            bool found = false;
            for(auto buf: allocated_buffers[size]) {
                if(used_buffers.contains(buf)) {
                    continue;
                }
                buffer = buf;
                found = true;
            }
            //if no buffer was found, create a new one
            if(!found) {
                buffer = AllocateBuffer(size);
            }
            used_buffers.insert(buffer);
            return buffer;
        }

        uint32_t GetRequiredAllocatedStorage() const {
            uint32_t total = 0;
            for(auto& [size, buffers]: allocated_buffers) {
                total += (uint32_t)size * (uint32_t)buffers.size();
            }
            return total;
        }

        ~BufferManager() {
            for(auto& [size, buffers]: allocated_buffers) {
                for(auto& buffer: buffers) {
                    delete buffer;
                }
            }
        }
    };

}  // namespace TensorFrost
