#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <stdexcept>

namespace TensorFrost {
using namespace std;
using uint = unsigned int;

inline uint AsUint(float f) { return *reinterpret_cast<uint*>(&f); }
inline uint AsUint(int i) { return *reinterpret_cast<uint*>(&i); }
inline float AsFloat(uint i) { return *reinterpret_cast<float*>(&i); }
inline float AsFloat(int i) { return *reinterpret_cast<float*>(&i); }
inline int AsInt(float f) { return *reinterpret_cast<int*>(&f); }
inline int AsInt(uint i) { return *reinterpret_cast<int*>(&i); }

int GetSize(const vector<int>& shape);

template <typename T, int N>
class FlagSet {
	array<int, N> data;

public:
	FlagSet() {
		data.fill(-1);
	}

	void set(T flag) {
		data[(int)flag] = 0;
	}

	template <typename... Args>
	void set(T flag, Args... args) {
		set(flag);
		set(args...);
	}

	void set(T flag, bool value) {
		data[(int)flag] = value ? 0 : -1;
	}

	void set(T flag, int value) {
		if(value < 0) {
			throw std::runtime_error("Flag data must be non-negative");
		}
		data[(int)flag] = value + 1;
	}

	void remove(T flag) {
		data[(int)flag] = -1;
	}

	template <typename... Args>
	void remove(T flag, Args... args) {
		remove(flag);
		remove(args...);
	}

	bool has(T flag) const {
		return data[(int)flag] != -1;
	}

	template <typename... Args>
	bool has(T flag, Args... args) const {
		return has(flag) && has(args...);
	}

	int get(T flag, bool throw_error = true) const {
		int res = data[(int)flag] - 1;
		if (throw_error && res < 0) {
			throw std::runtime_error("Flag data is not set");
		}
		return res;
	}

	void copy_all_given(const FlagSet<T, N>& other, unordered_set<T> only) {
		for (int i = 0; i < N; i++) {
			data[i] = -1;
			T flag = (T)i;
			if (only.contains(flag)) {
				data[(int)flag] = other.data[(int)flag];
			}
		}
	}

	void copy_all_except(const FlagSet<T, N>& other, unordered_set<T> except) {
		for (int i = 0; i < N; i++) {
			data[i] = -1;
			T flag = (T)i;
			if (!except.contains(flag)) {
				data[(int)flag] = other.data[(int)flag];
			}
		}
	}

	void copy_all(const FlagSet<T, N>& other) {
		for (int i = 0; i < N; i++) {
			data[i] = other.data[i];
		}
	}

	size_t count() const {
		size_t res = 0;
		for (int i = 0; i < N; i++) {
			if (data[i] != -1) {
				res++;
			}
		}
		return res;
	}

	unordered_map<T, int> get_data() const {
		unordered_map<T, int> res;
		for (int i = 0; i < N; i++) {
			T flag = (T)i;
			if (has(flag)) {
				res[flag] = data[i] - 1;
			}
		}
		return res;
	}
};

}  // namespace TensorFrost