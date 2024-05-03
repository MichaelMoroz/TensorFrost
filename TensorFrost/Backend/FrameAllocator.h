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

class Frame {
 public:
	uint32_t start;
	uint32_t size;
	uint32_t end;
};

class FrameAllocator {
 private:
	std::map<uint32_t, Frame*> Frames_;
	const uint32_t min_frame_size = 4;

 public:
	FrameAllocator() = default;

	// Frame iterator
	using iterator = typename std::map<uint32_t, Frame*>::iterator;
	iterator begin() { return Frames_.begin(); }
	iterator end() { return Frames_.end(); }

	Frame* AllocateFrame(uint32_t size) {
		size = std::max(size, min_frame_size);

		uint32_t start = 0;
		for (auto& pair : Frames_) {
			Frame* frame = pair.second;
			if (frame->start - start >= size) {
				// Found a free available slot
				break;
			}
			start = frame->end;
		}

		auto* new_frame = new Frame();
		new_frame->start = start;
		new_frame->size = size;
		new_frame->end = start + size;
		Frames_[start] = new_frame;

		return new_frame;
	}

	void FreeFrame(Frame frame) {
		auto it = Frames_.find(frame.start);
		if (it != Frames_.end()) {
			Frames_.erase(it);
		}
	}

	uint32_t GetRequiredAllocatedStorage() const {
		if (Frames_.empty()) return 0;
		// Get the end of the last frame in the sorted map
		return std::prev(Frames_.end())->second->end;
	}
};

}  // namespace TensorFrost
