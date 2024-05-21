#include "BufferManager.h"

namespace TensorFrost {
    void BufferManager::DeallocateBuffer(Buffer *buffer) {
        used_buffers.erase(buffer);
    }

    void BufferManager::RemoveBuffer(Buffer *buffer) {
        for(auto& [size, buffers]: allocated_buffers) {
            buffers.erase(buffer);
        }
        DeallocateBuffer(buffer);
        delete buffer;
    }

    Buffer * BufferManager::TryAllocateBuffer(int size) {
        //try to find a non-used buffer of the correct size
        Buffer* buffer = nullptr;
        bool found = false;
        //find the smallest buffer that is larger than the requested size
        int min_size = size;
        int max_size = 32 * size;
        //get iterator to the first buffer that is larger than the requested size
        auto it = allocated_buffers.lower_bound(min_size);
        //if no buffer is larger than the requested size, get the first buffer
        if(it == allocated_buffers.end()) {
            it = allocated_buffers.begin();
        }
        //iterate through the buffers
        for(; it != allocated_buffers.end(); it++) {
            if(it->first < size) {
                continue;
            }
            for(auto buf: it->second) {
                if(used_buffers.contains(buf)) {
                    continue;
                }
                buffer = buf;
                found = true;
            }
            if(found) {
                break;
            }
        }
        //if no buffer was found, create a new one
        if(!found) {
            buffer = AllocateBuffer(size);
        }
        used_buffers.insert(buffer);
        return buffer;
    }

    uint32_t BufferManager::GetRequiredAllocatedStorage() const {
        uint32_t total = 0;
        for(auto& [size, buffers]: allocated_buffers) {
            total += (uint32_t)size * (uint32_t)buffers.size();
        }
        return total;
    }

    uint32_t BufferManager::GetUnusedAllocatedStorage() const {
        uint32_t total = 0;
        for(auto& [size, buffers]: allocated_buffers) {
            for(auto& buffer: buffers) {
                if(!used_buffers.contains(buffer)) {
                    total += size;
                }
            }
        }
        return total;
    }

    BufferManager::~BufferManager() {
        for(auto& [size, buffers]: allocated_buffers) {
            for(auto& buffer: buffers) {
                delete buffer;
            }
        }
    }
}  // namespace TensorFrost