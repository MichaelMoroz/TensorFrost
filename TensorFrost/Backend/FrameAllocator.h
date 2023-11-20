#pragma once

#include <functional>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>

namespace TensorFrost {

using namespace std;

class Frame {
public:
    uint32_t start;
    uint32_t size;
    uint32_t end;
};

class FrameAllocator {
private:
    std::map<uint32_t, Frame*> Frames;

public:
    FrameAllocator() {}

    // Frame iterator
    typedef typename std::map<uint32_t, Frame*>::iterator iterator;
    iterator begin() { return Frames.begin(); }
    iterator end() { return Frames.end(); }

    Frame* AllocateFrame(uint32_t size) {
        uint32_t start = 0;
        for (auto& pair : Frames) {
            Frame* frame = pair.second;
            if (frame->start - start >= size) {
                // Found a free available slot
                break;
            }
			start = frame->end;
        }

        Frame* newFrame = new Frame();
        newFrame->start = start;
        newFrame->size = size;
        newFrame->end = start + size;
        Frames[start] = newFrame;

        return newFrame;
    }

    void FreeFrame(Frame frame) {
        auto it = Frames.find(frame.start);
        if (it != Frames.end()) {
            Frames.erase(it);
        }
    }

    uint32_t GetRequiredAllocatedStorage() const {
        if (Frames.empty()) return 0;
        // Get the end of the last frame in the sorted map
		 return std::prev(Frames.end())->second->end;
    }
};

}  // namespace TensorFrost