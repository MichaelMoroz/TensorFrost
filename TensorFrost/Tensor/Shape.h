#pragma once

#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

namespace TensorFrost {
class Tensor;

enum DimensionType {
	// Tensor dimensions
	TThread,
	TLoop,

	// Compute dimensions (for parallelism)
	COuterLoop,  // outermost loop
	CBlock,      // workgroups
	CGroup,      // workitems
	CInnerLoop,  // innermost loop (per workitem)
};

class Dimension {
 public:
	int size = 1;       // -1 means unknown
	int min_size = -1;  // -1 means no minimum
	int max_size = -1;  // -1 means no maximum

	DimensionType type = DimensionType::TThread;

	Dimension(int size, DimensionType type = DimensionType::TThread) {
		this->size = size;
		this->type = type;
	}

	Dimension(int min_size, int max_size,
	          DimensionType type = DimensionType::TThread) {
		this->size = -1;
		this->min_size = min_size;
		this->max_size = max_size;
		this->type = type;
	}
};

class Shape {
 public:
	std::vector<Dimension> dimensions;

	explicit Shape(std::vector<Dimension> dimensions) {
		this->dimensions = std::move(dimensions);
	}

	template <typename... Args>
	explicit Shape(int size, const Args&... args) {
		dimensions.push_back(Dimension());
		dimensions[0].size = size;
		AddDimensions(args...);
	}

	template <typename... Args>
	void AddDimensions(const Dimension& dimension, const Args&... args) {
		dimensions.push_back(dimension);
		AddDimensions(args...);
	}

	Shape() { dimensions.push_back(Dimension(1)); }

	Shape(std::vector<int> shape) {
		for (auto size : shape) {
			dimensions.push_back(Dimension(size));
		}
	}

	int size() const { return dimensions.size(); }

	[[nodiscard]] int GetSize() const {
		int size = 1;
		for (auto dimension : dimensions) {
			size *= dimension.size;
		}
		return size;
	}

	std::vector<int> GetShape() const {
		std::vector<int> shape = std::vector<int>();
		for (auto& dimension : dimensions) {
			shape.push_back(dimension.size);
		}
		return shape;
	}

	int operator[](int index) const {
		if (index < dimensions.size()) {
			return dimensions[index].size;
		}
		return 1;
	}
};
}  // namespace TensorFrost