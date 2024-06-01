#include "TensorMemory.h"

namespace TensorFrost {

size_t GetLinearSize(const vector<size_t>& shape) {
	size_t size = 1;
	for (size_t dim : shape) {
		size *= dim;
	}
	return size;
}

vector<size_t> GetShape(const TFTensor *tensor) {
	vector<size_t> shape;
	for (size_t i = 0; i < tensor->dim; i++) {
		shape.push_back(tensor->shape[i]);
	}
	return shape;
}

size_t GetSize(const TFTensor *tensor) {
	size_t size = 1;
	for (size_t i = 0; i < tensor->dim; i++) {
		size *= tensor->shape[i];
	}
	return size;
}

void TensorMemoryManager::DeallocateBuffer(TFBuffer *buffer) {
    used_buffers.erase(buffer);
    unused_time[buffer] = 0;
}

void TensorMemoryManager::RemoveBuffer(TFBuffer *buffer) {
    if(!buffers_to_delete.contains(buffer)) {
        throw std::runtime_error("Buffer not marked for deletion");
    }
    size_t size = buffer->size;
    allocated_buffers[size].erase(buffer);
    unused_time.erase(buffer);
    delete buffer;
}

void TensorMemoryManager::UpdateTick() {
    //increment the unused time of all buffers
    for(auto& [buffer, time]: unused_time) {
        if(time > MAX_UNUSED_TIME) {
            buffers_to_delete.insert(buffer);
        } else {
            unused_time[buffer] = time + 1;
        }
    }
}

TFBuffer *TensorMemoryManager::TryAllocateBuffer(size_t size) {
    UpdateTick();
    //try to find a non-used buffer of the correct size
    TFBuffer* buffer = nullptr;
    bool found = false;
    //find the smallest buffer that is larger than the requested size
    size_t min_size = size;
    size_t max_size = 8 * size;
    //get iterator to the first buffer that is larger than the requested size
    auto it = allocated_buffers.lower_bound(min_size);
    //if no buffer is larger than the requested size, get the first buffer
    if(it == allocated_buffers.end()) {
        it = allocated_buffers.begin();
    }
    //iterate through the buffers
    for(; it != allocated_buffers.end(); it++) {
        if(it->first > max_size) {
            break;
        }
        if(it->first < size) {
            continue;
        }
        for(auto buf: it->second) {
            if(used_buffers.contains(buf) && !buffers_to_delete.contains(buf)) {
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
    else {
        unused_time.erase(buffer);
    }
    used_buffers.insert(buffer);
    return buffer;
}

size_t TensorMemoryManager::GetRequiredAllocatedStorage() const {
    size_t total = 0;
    for(auto& [size, buffers]: allocated_buffers) {
        total += (uint32_t)size * (uint32_t)buffers.size();
    }
    return total;
}

size_t TensorMemoryManager::GetUnusedAllocatedStorage() const {
    size_t total = 0;
    for(auto& [size, buffers]: allocated_buffers) {
        for(auto& buffer: buffers) {
            if(!used_buffers.contains(buffer)) {
                total += size;
            }
        }
    }
    return total;
}

TensorMemoryManager::~TensorMemoryManager() {
    for(auto& [size, buffers]: allocated_buffers) {
        for(auto& buffer: buffers) {
            delete buffer;
        }
    }
}

TensorMemoryManager* global_memory_manager = nullptr;

}  // namespace TensorFrost