#pragma once

#include <vector>

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

}  // namespace TensorFrost