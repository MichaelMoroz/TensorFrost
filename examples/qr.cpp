#include <cmath>
#include <omp.h>
#include <initializer_list>

typedef unsigned int uint;

inline int min(int a, int b)
{
  return a < b ? a : b;
}

inline int max(int a, int b)
{
  return a > b ? a : b;
}

inline float min(float a, float b)
{
  return a < b ? a : b;
}

inline float max(float a, float b)
{
  return a > b ? a : b;
}

inline float asfloat(uint x)
{
  return *(float*)&x;
}

inline uint asuint(float x)
{
  return *(uint*)&x;
}

inline int clamp(int x, int a, int b)
{
  return min(max(x, a), b);
}

inline float clamp(float x, float a, float b)
{
  return min(max(x, a), b);
}

inline void InterlockedAdd(int* memory, int address, int value)
{
  #pragma omp atomic
  memory[address] += value;
}

inline void InterlockedAdd(uint* memory, int address, uint value)
{
  #pragma omp atomic
  memory[address] += value;
}

inline void InterlockedAdd(float* memory, int address, float value)
{
  #pragma omp atomic
  memory[address] += value;
}

inline void InterlockedAnd(int* memory, int address, int value)
{
  #pragma omp atomic
  memory[address] &= value;
}

inline void InterlockedAnd(uint* memory, int address, uint value)
{
  #pragma omp atomic
  memory[address] &= value;
}

inline void InterlockedOr(int* memory, int address, int value)
{
  #pragma omp atomic
  memory[address] |= value;
}

inline void InterlockedOr(uint* memory, int address, uint value)
{
  #pragma omp atomic
  memory[address] |= value;
}

inline void InterlockedXor(int* memory, int address, int value)
{
  #pragma omp atomic
  memory[address] ^= value;
}

inline void InterlockedXor(uint* memory, int address, uint value)
{
  #pragma omp atomic
  memory[address] ^= value;
}

//initialize R and Q
extern "C" __declspec(dllexport) void kernel_0(uint* var, uint* off, uint* mem, uint* shape)
{
  #pragma omp parallel for shared(mem) 
  for (int dim0 = 0; dim0 < shape[0]; dim0++)
  for (int dim1 = 0; dim1 < shape[1]; dim1++)
  {
    float v0 = 0.000000f;
    mem[off[0] + ((clamp(dim0, 0, 63) * 64) + clamp(dim1, 0, 63))] = asuint(v0);
    float v2 = 0.000000f;
    mem[off[1] + ((clamp(dim0, 0, 63) * 64) + clamp(dim1, 0, 63))] = asuint(v2);
  }
}

//something (probably just copy column)
extern "C" __declspec(dllexport) void kernel_1(uint* var, uint* off, uint* mem, uint* shape)
{
  #pragma omp parallel for shared(mem) 
  for (int dim0 = 0; dim0 < shape[0]; dim0++)
  {
    float v0 = asfloat(mem[off[0] + ((clamp(dim0, 0, var[1] - 1) * (int)var[0]) + clamp(0, 0, var[0] - 1))]);
    mem[off[1] + clamp(dim0, 0, var[0])] = asuint(v0);
  }
}

//zero out the sum counter
extern "C" __declspec(dllexport) void kernel_2(uint* var, uint* off, uint* mem, uint* shape)
{
  #pragma omp parallel for shared(mem) 
  for (int dim0 = 0; dim0 < shape[0]; dim0++)
  {
    float v0 = 0.000000f;
    mem[off[0] + clamp(dim0, 0, 0)] = asuint(v0);
  }
}

//sum the squares of the first column
extern "C" __declspec(dllexport) void kernel_3(uint* var, uint* off, uint* mem, uint* shape)
{
  #pragma omp parallel for shared(mem) 
  for (int dim0 = 0; dim0 < shape[0]; dim0++)
  {
    float v0 = pow(asfloat(mem[off[0] + clamp(dim0, 0, var[0]-1)]), 2.000000f);
    InterlockedAdd((float*)mem, off[1] + clamp(0, 0, 0), v0);
  }
}

//take the square root of the sum and store it in R[0,0]
extern "C" __declspec(dllexport) void kernel_4(uint* var, uint* off, uint* mem, uint* shape)
{
  #pragma omp parallel for shared(mem) 
  for (int dim0 = 0; dim0 < shape[0]; dim0++)
  {
    mem[off[1] + (clamp(0, 0, var[1]-1) * (int)var[0]) + clamp(0, 0, var[0]-1)] = asuint(sqrt(asfloat(mem[off[0]])));
  }
}

//divide the first column by the square root of the sum and store it in Q[i,0]
extern "C" __declspec(dllexport) void kernel_5(uint* var, uint* off, uint* mem, uint* shape)
{
  #pragma omp parallel for shared(mem) 
  for (int dim0 = 0; dim0 < shape[0]; dim0++)
  {
    float v0 = asfloat(mem[off[0] + ((clamp(dim0, 0, 63) * 64) + clamp(0, 0, 63))]) / asfloat(mem[off[1] + ((clamp(0, 0, 63) * 64) + clamp(0, 0, 63))]);
    mem[off[2] + ((clamp(dim0, 0, 63) * 64) + clamp(0, 0, 63))] = asuint(v0);
  }
}

//compute the outer product of the Q and A 
extern "C" __declspec(dllexport) void kernel_6(uint* var, uint* off, uint* mem, uint* shape)
{
  #pragma omp parallel for shared(mem) 
  for (int dim0 = 0; dim0 < shape[0]; dim0++)
  for (int dim1 = 0; dim1 < shape[1]; dim1++)
  {
    float v0 = asfloat(mem[off[0] + ((clamp(dim0, 0, 63) * 64) + clamp(0, 0, 63))]) * asfloat(mem[off[1] + ((clamp(dim0, 0, 63) * 64) + clamp((dim1 + 1), 0, 63))]);
    mem[off[2] + ((clamp(dim0, 0, 63) * 63) + clamp(dim1, 0, 62))] = asuint(v0);
  }
}

//zero out the sum counter
extern "C" __declspec(dllexport) void kernel_7(uint* var, uint* off, uint* mem, uint* shape)
{
  #pragma omp parallel for shared(mem) 
  for (int dim0 = 0; dim0 < shape[0]; dim0++)
  {
    float v0 = 0.000000f;
    mem[off[0] + clamp(dim0, 0, 62)] = asuint(v0);
  }
}

//compute dot products of the columns of the outer product and store them in R[i,j]
extern "C" __declspec(dllexport) void kernel_8(uint* var, uint* off, uint* mem, uint* shape)
{
  #pragma omp parallel for shared(mem) 
  for (int dim0 = 0; dim0 < shape[0]; dim0++)
  for (int dim1 = 0; dim1 < shape[1]; dim1++)
  {
    float v0 = asfloat(mem[off[0] + ((clamp(dim0, 0, 63) * 63) + clamp(dim1, 0, 62))]);
    InterlockedAdd((float*)mem, off[1] + clamp(dim1, 0, 62), v0);
  }
}

//store the dot products in R[i,j]
extern "C" __declspec(dllexport) void kernel_9(uint* var, uint* off, uint* mem, uint* shape)
{
  #pragma omp parallel for shared(mem) 
  for (int dim0 = 0; dim0 < shape[0]; dim0++)
  {
    mem[off[1] + ((clamp(0, 0, 63) * 64) + clamp((dim0 + 1), 0, 63))] = asuint(asfloat(mem[off[0] + clamp(dim0, 0, 62)]));
  }
}

//subtract the outer product of Q and R from A
extern "C" __declspec(dllexport) void kernel_10(uint* var, uint* off, uint* mem, uint* shape)
{
  #pragma omp parallel for shared(mem) 
  for (int dim0 = 0; dim0 < shape[0]; dim0++)
  for (int dim1 = 0; dim1 < shape[1]; dim1++)
  {
    int v0 = dim1 + 1;
    float v1 = asfloat(mem[off[0] + ((clamp(dim0, 0, 63) * 64) + clamp(0, 0, 63))]) * asfloat(mem[off[1] + ((clamp(0, 0, 63) * 64) + clamp(v0, 0, 63))]);
    float v2 = asfloat(mem[off[2] + ((clamp(dim0, 0, 63) * 64) + clamp(v0, 0, 63))]) - v1;
    mem[off[2] + ((clamp(dim0, 0, 63) * 64) + clamp(v0, 0, 63))] = asuint(v2);
  }
}

void dispatch(void(*kernel)(uint*, uint*, uint*, uint*), uint* mem, std::initializer_list<int> off, std::initializer_list<int> var, std::initializer_list<int> shape)
{
  uint off_arr[off.size()];
  uint var_arr[var.size()];
  uint shape_arr[shape.size()];

  for (int i = 0; i < off.size(); i++)
  {
    off_arr[i] = *(off.begin() + i);
  }

  for (int i = 0; i < var.size(); i++)
  {
    var_arr[i] = *(var.begin() + i);
  }

  for (int i = 0; i < shape.size(); i++)
  {
    shape_arr[i] = *(shape.begin() + i);
  }

  kernel(var_arr, off_arr, mem, shape_arr);
} 

extern "C" __declspec(dllexport) void program(uint* in, uint* out, uint* mem, uint(*allocate)(uint), uint(*deallocate)(uint))
{
  //allocate memory for R
  int n = mem[in[0] + clamp(0, 0, 0)];
  int m = mem[in[1] + clamp(0, 0, 0)];
  int var0 = n * m;
  int off0 = allocate(var0);
  //allocate memory for Q
  int var1 = n * m;
  int off1 = allocate(var1);

  //initialize R and Q
  dispatch(kernel_0, mem, {off0, off1}, {n, m}, {n, m});

  //dealocate memory for R
  deallocate(off0);

  //output
  out[0] = off1;  
}