# 🥶 TensorFrost 🥶
Yet another Python tensor library with autodifferentiation (TODO). Currently very much a work in progress.

The main idea of this library is to compile optimal fused kernels for the GPU given a set of numpy-ish functions, including other more complex things like loops and conditionals (also TODO)

Currently working platforms:
| Backend/OS | CPU | CUDA | Vulkan |
|------------|-----|------|--------|
| Windows    | 🚧   | ⛔    | ⛔      |
| Linux      | ⛔   | ⛔    | ⛔      |

## Examples

<a href="examples/wave_simulation.ipynb"><img src="examples/sin_gordon.gif" height="192px"></a>
<a href="examples/fluid_simulation.ipynb"><img src="examples/fluid.gif" height="192px"></a>

## Installation
TensorFrost is currently not available on PyPI, so you'll have to install it from source. 
You need to have CMake installed, as well as a C++ compiler that supports C++17 and Python 3.7 or higher.

First clone the repository:
```bash
git clone --recurse-submodules https://github.com/MichaelMoroz/TensorFrost.git
cd TensorFrost
```

Then run cmake to build the library:
```bash
cmake -S . -B build && cmake --build build
```

The cmake script will automatically install the compiled python module into your python environment.

## Usage

### Setup
First you need to import the library:
```python
import TensorFrost as tf
```

Then you need to initialize the library with the device you want to use:
```python
tf.initialize(tf.cpu, "cl_compile.bat /O2 /fp:fast /openmp") # Windows + MSVC (currently the only working compiler out of the box)
```
You must yourself provide a compiler path here. Currently only MSVC is supported out of the box.
The cl_compile.bat script looks like this for example:

```bat
@echo off
call "Path\To\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cl %*
```

### Basic usage

Now you can create and compile functions, for example here is a very simple function does a wave simulation:
```python
def WaveEq():
    u = tf.input([-1, -1], tf.float32)
    v = tf.input([-1, -1], tf.float32)

    i,j = u.indices
    laplacian = u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - u * 4.0
    v_new = v + dt*laplacian
    u_new = u + dt*v_new

    return [v_new, u_new]

wave_eq = tf.program(WaveEq)
```

The tensor programs take and output tensor memory buffers, which can be created from numpy arrays:
```python
A = tf.memory(np.zeros([100, 100], dtype=np.float32))
B = tf.memory(np.zeros([100, 100], dtype=np.float32))
```

Then you can run the program:
```python
A, B = wave_eq(A, B)
```

To get the result back into a numpy array, you can use the `numpy` property:
```python
Anp = A.numpy
```

TensorFrost does not support JIT compilation (currently no plans either), so you must create the program before running it. Therefore the tensor operations must only be used inside a tensor program. Operations outside the function will throw an error, so if you want to do operations outside you must read the data into a numpy array first.

### Operations

TensorFrost supports most of the basic numpy operations, including indexing, arithmetic, and broadcasting (only partially for now).
The core operation is the indexing operation, which is used to specify indices for accessing the tensor data. Depending on the dimensinality of the tensor there can be N indices. This operation is similar to numpy's `np.ogrid` and `np.mgrid` functions, but it is basically free due to fusion.

```python
#can be created either from a provided shape or from a tensor
i,j = tf.indices([8, 8]) 
i,j = A.indices
```

For example `i` contains:

```
[[0, 0, 0, ..., 0, 0, 0],
 [1, 1, 1, ..., 1, 1, 1],
 [2, 2, 2, ..., 2, 2, 2],
    ...,
 [7, 7, 7, ..., 7, 7, 7]]
```

And analogously for `j`.

These indices can then be used to index into the tensor data, to either read or write data:
```python
#set elements [16:32, 16:32] to 1.0
i,j = tf.indices([16, 16]) 
B[i+16, j+16] = 1.0

#read elements [8:24, 8:24]
i,j = tf.indices([16, 16])
C = B[i+8, j+8]
```

Here we can see that the shape of the "computation" is not the same as the shape of the tensor, and one thread is spawned for each given index. This is the main idea of TensorFrost. Then all sequential computations of the same shape are fused into a single kernel, if they are not dependent on each other in a non-trivial way.

When doing out-of-bounds indexing, the index is currently clamped to the tensor shape. This is not ideal, but it is the simplest way to handle this. In the future there will be a way to specify the boundary conditions.

### Scatter operations

These operations allow implementing non-trivial reduction operations, including, for example, matrix multiplication:

```python
def MatrixMultiplication():
    A = tf.input([-1, -1], tf.float32)
    B = tf.input([-1, -1], tf.float32)

    N, M = A.shape
    K = B.shape[1]

    i, j, k = tf.indices([N, K, M])

    C = tf.zeros([N, K])
    tf.scatterAdd(C[i, j], A[i, k] * B[k, j])

    return [C]

matmul = tf.program(MatrixMultiplication)
```

Here the 3D nature of the matrix multiplication is apparent. The scatter operation is used to accumulate the results of the row-column dot products into the elements of the resulting matrix.

(This is not the most efficient way to implement matrix multiplication, but it is the simplest way to show how scatter operations work. In the future though, scatters will have the ability to optimize into loops if possible.)

### Loops and conditionals

TODO

### Autodifferentiation

TODO

### Advanced usage

TODO


## Roadmap 

Core features:
- [x] Basic operations (memory, indexing, arithmetic, etc.)
- [x] Basic kernel fusion and compilation
- [ ] Advanced built-in functions (random, special functions, etc.)
- [ ] Advanced operations (loops, conditionals, etc.)
- [ ] Autodifferentiation
- [ ] Kernel code and execution graph export and editing
- [ ] Advanced data types and quantization
- [ ] Advanced IR optimizations
- [ ] Kernel shape and cache optimization
  
Algorithm library:
- [ ] Sort, scan, reduction, etc.
- [ ] Matrix operations (matrix multiplication, etc.)
- [ ] Advanced matrix operations (QR, SVD, eigenvalues, etc.)
- [ ] Fast Fourier Transform
- [ ] High-level neural network layers (convolution, etc.)

Platforms:
- [x] Windows
- [ ] Linux
- [ ] MacOS

Backends:
- [x] CPU (using user-provided compiler)
- [ ] ISPC (for better CPU utilization)
- [ ] Vulkan
- [ ] CUDA
- [ ] WGPU (for web)

(hopefully im not going to abandon this project before finishing lol)

## Contributing
Contributions are welcome! If you want to contribute, please open an issue first to discuss the changes you want to make.