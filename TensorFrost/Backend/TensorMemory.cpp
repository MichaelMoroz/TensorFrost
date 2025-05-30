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

TFBuffer * TensorMemoryManager::AllocateBuffer(size_t size) {
    TFBuffer* buffer = CreateBuffer(size);
    if(allocation_history.contains(size)) {
        size_t old_delay = GetDeallocationDelay(size);
        allocation_delay[size] = std::min(std::max(old_delay, tick - allocation_history[size]), MAX_POSSIBLE_UNUSED_TIME);
    } else {
        allocation_delay[size] = DEFAULT_MAX_UNUSED_TIME;
    }
    buffers_created++;
    //add the buffer to the list of allocated buffers
    allocated_buffers[size].insert(buffer);
    allocation_history[size] = tick;
    return buffer;
}

TFTensor * TensorMemoryManager::AllocateTensor(const vector<size_t> &shape, const TFDataFormat type, const char* name) {
    size_t size = GetLinearSize(shape);

    if (size == 0) {
        throw invalid_argument("Trying to allocate a tensor with size 0");
    }

    TFBuffer* buf = TryAllocateBuffer(size);
    buf->read_only = false;
    ((TFBufferTemplate*)buf)->UpdateName(name);
    return MakeTensor(shape, buf, type);
}

TFTensor * TensorMemoryManager::AllocateTensorWithData(const vector<size_t> &shape, const vector<uint32_t> &data,
    const TFDataFormat type, bool read_only, const char* name) {
    TFTensor* tensor_memory = AllocateTensor(shape, type, name);
    tensor_memory->buffer->read_only = read_only;
    ((TFBufferTemplate*)tensor_memory->buffer)->SetDataAtOffset(0, data);
    return tensor_memory;
}

void TensorMemoryManager::DeallocateTensor(TFTensor tensor) {
    DeallocateBuffer(tensor.buffer);
}

TFTensor * TensorMemoryManager::MakeTensor(size_t *shape, size_t dim, TFBuffer *buf, TFDataFormat type) {
    TFTensor* tensor = new TFTensor();
    tensor->buffer = buf;
    tensor->dim = dim;
    tensor->shape = shape;
    tensor->format = type;
    return tensor;
}

TFTensor * TensorMemoryManager::MakeTensor(const vector<size_t> &shape, TFBuffer *buf, TFDataFormat type) {
    size_t* shape_arr = new size_t[shape.size()];
    std::copy(shape.begin(), shape.end(), shape_arr);
    return MakeTensor(shape_arr, shape.size(), buf, type);
}

size_t TensorMemoryManager::GetAllocatedSize() const {
    size_t total = 0;
    for(auto& [size, buffers]: allocated_buffers) {
        total += size * buffers.size();
    }
    return total;
}

size_t TensorMemoryManager::GetUnusedAllocatedSize() const {
    size_t total = 0;
    for(auto& [size, buffers]: allocated_buffers) {
        for(auto& buffer: buffers) {
            if(unused_buffers.contains(buffer)) {
                total += size;
            }
        }
    }
    return total;
}

void TensorMemoryManager::DeallocateBuffer(TFBuffer *buffer) {
    unused_buffers.insert(buffer);
    buffer->time_since_used = 0;
    buffer->used_size = 0;
    buffer->up_to_date = false;
    buffer->name = "none";
}

void TensorMemoryManager::RemoveBuffer(TFBuffer *buffer) {
    size_t size = buffer->size;
    allocated_buffers[size].erase(buffer);
    unused_buffers.erase(buffer);
    DeleteBuffer(buffer);
    buffers_removed++;
}

//#define READBACK_DEBUG

vector<uint32_t> TensorMemoryManager::Readback(const TFTensor *memory) {
    vector<uint32_t> data(GetSize(memory));
#ifdef READBACK_DEBUG
    cout << "Reading back " << data.size() << " elements from buffer of size " << memory->buffer->size << endl;
#endif
    ((TFBufferTemplate*)memory->buffer)->GetDataAtOffset(0, data.size(), data.data());
    return data;
}

uint TensorMemoryManager::ReadbackValue(const TFTensor *memory, size_t index) {
    uint32_t data;
    ((TFBufferTemplate*)memory->buffer)->GetDataAtOffset(index, 1, &data);
    return data;
}

void TensorMemoryManager::Writeback(const TFTensor *memory, const vector<uint32_t> &data) {
    ((TFBufferTemplate*)memory->buffer)->SetDataAtOffset(0, data);
}

void TensorMemoryManager::WritebackValue(const TFTensor *memory, size_t index, uint32_t value) {
    ((TFBufferTemplate*)memory->buffer)->SetDataAtOffset(index, {value});
}

void TensorMemoryManager::UpdateTick() {
    unordered_set<TFBuffer*> buffers_to_delete;

    for(auto& buffer: unused_buffers) {
        size_t buf_size = buffer->size;
        if(buffer->time_since_used > GetDeallocationDelay(buf_size)) {
            buffers_to_delete.insert(buffer);
        } else {
            buffer->time_since_used++;
        }
    }

    //delete all buffers that are marked for deletion
    for(auto& buffer: buffers_to_delete) {
        RemoveBuffer(buffer);
    }
    tick++;

#ifdef DEBUG_DYNAMIC_ALLOCATION
    if(tick%2048 == 0) {
        if(buffers_created> 0 || buffers_removed > 0) {
            cout << "Note: " << buffers_created << " buffers created and " << buffers_removed << " buffers removed in the last 2048 ticks" << endl;
        }
        buffers_created = 0;
        buffers_removed = 0;
    }
#endif
}

size_t TensorMemoryManager::GetDeallocationDelay(size_t buf_size) const {
    if(allocation_delay.contains(buf_size)) {
        return allocation_delay.at(buf_size);
    }

    return DEFAULT_MAX_UNUSED_TIME;
}

TFBuffer *TensorMemoryManager::TryAllocateBuffer(size_t size) {
    //try to find a non-used buffer of the correct size
    TFBuffer* buffer = nullptr;
    bool found = false;
    //find the smallest buffer that is larger than the requested size
    size_t min_size = size;
    size_t max_size = 16 * size;
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
            if(!unused_buffers.contains(buf)) {
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
    } else {
        unused_buffers.erase(buffer);
    }
    buffer->used_size = size;
    buffer->time_since_used = 0;
    UpdateTick();
    return buffer;
}

TensorMemoryManager::~TensorMemoryManager() {
    for(auto& [size, buffers]: allocated_buffers) {
        for(auto& buffer: buffers) {
            DeleteBuffer(buffer);
        }
    }
}

TensorMemoryManager* global_memory_manager = nullptr;

}  // namespace TensorFrost