#pragma once

#include <vector>
#include <bitset>
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

template <typename T, int N, int MV = 8>
class FlagSet {
	bitset<N> flags;
	array<array<int, MV>, N> data;

public:
	FlagSet() {
		flags.reset();
		for (int i = 0; i < N; i++) {
			data[i].fill(-1);
		}
	}

	void set(T flag) {
		flags.set((int)flag);
	}

	template <typename... Args>
	void set(T flag, Args... args) {
		set(flag);
		set(args...);
	}

	void set(T flag, bool value) {
		flags.set((int)flag, value);
	}

	void set(T flag, int value, int index = 0) {
		flags.set((int)flag);
		if (index >= MV) {
			throw std::runtime_error("Index out of bounds");
		}
		data[(int)flag][index] = value;
	}

	void remove(T flag) {
		flags.reset((int)flag);
	}

	template <typename... Args>
	void remove(T flag, Args... args) {
		remove(flag);
		remove(args...);
	}

	bool has(T flag) const {
		return flags.test((int)flag);
	}

	template <typename... Args>
	bool has(T flag, Args... args) const {
		return has(flag) && has(args...);
	}

	int get(T flag, int index = 0, bool throw_error = true) const {
		int res = data[(int)flag][index];
		if (throw_error && res == -1) {
			throw std::runtime_error("Flag not found");
		}
		return res;
	}

	void copy_all_given(const FlagSet<T, N>& other, unordered_set<T> only) {
		flags.reset();
		for (T flag : only) {
			if (other.has(flag)) {
				flags.set((int)flag);
				data[(int)flag] = other.data[(int)flag];
			}
		}
	}

	void copy_all_except(const FlagSet<T, N>& other, unordered_set<T> except) {
		flags.reset();
		for (int i = 0; i < N; i++) {
			T flag = (T)i;
			if (other.has(flag) && !except.contains(flag)) {
				flags.set((int)flag);
				data[(int)flag] = other.data[(int)flag];
			}
		}
	}

	void copy_all(const FlagSet<T, N>& other) {
		flags = other.flags;
		for (int i = 0; i < N; i++) {
			data[i] = other.data[i];
		}
	}

	size_t count() const {
		return flags.count();
	}

	unordered_map<T, map<int, int>> get_data() {
		unordered_map<T, map<int, int>> res;
		for (int i = 0; i < N; i++) {
			T flag = (T)i;
			if (has(flag)) {
				map<int, int> flag_data;
				for (int j = 0; j < MV; j++) {
					if (data[i][j] != -1) {
						flag_data[j] = data[i][j];
					}
				}
				res[flag] = flag_data;
			}
		}
		return res;
	}
};

}  // namespace TensorFrost