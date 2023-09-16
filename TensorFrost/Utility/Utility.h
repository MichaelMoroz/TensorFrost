#pragma once

#include <intrin.h>

namespace TensorFrost {
using namespace std;
typedef unsigned int uint;

// function to convert float bits to uint bits
inline uint AsUint(float f) { return *reinterpret_cast<uint*>(&f); }

// function to convert uint bits to float bits
inline float AsFloat(uint i) { return *reinterpret_cast<float*>(&i); }

// function to convert float to int
inline int AsInt(float f) { return *reinterpret_cast<int*>(&f); }

// function to convert int to float
inline float AsFloat(int i) { return *reinterpret_cast<float*>(&i); }
}  // namespace TensorFrost