# 🥶 TensorFrost (v0.1.2) 🥶
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

## From PyPI

```bash
pip install tensorfrost
```

## From source

You need to have CMake installed to build the library. 

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

### Building wheel packages (optional)

You can either call `clean_rebuild.bat %PYTHON_VERSION%` to build the wheel packages for the specified python version (the version needs to be installed beforehand), or you can build them for all versions by calling `build_all_python_versions.bat`. The scripts will automatically build and install the library for each python version, and then build the wheel packages to the `PythonBuild/dist` folder.

## Usage

### Setup
For the library to work you need a C++ compiler that supports C++17 (Currently only Microsoft Visual Studio Compiler).

First you need to import the library:
```python
import TensorFrost as tf
```

Then you need to initialize the library with the device you want to use and the kernel compiler flags (different for each platform):
```python
tf.initialize(tf.cpu, "/O2 /fp:fast /openmp") # Windows + MSVC (currently the only working compiler out of the box)
```

TensorFrost will find any available MSVC installation and use it to compile the kernels. If you want to use a different compiler, you can specify the path to the compiler executable (TODO).

### Basic usage

Now you can create and compile functions, for example here is a very simple function does a wave simulation:
```python
def WaveEq():
    #shape is not specified -> shape is inferred from the input tensor (can result in slower execution)
    u = tf.input([-1, -1], tf.float32)
    #shape must match 
    v = tf.input(u.shape, tf.float32)

    i,j = u.indices
    laplacian = u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - u * 4.0
    v_new = v + dt*laplacian
    u_new = u + dt*v_new

    return [v_new, u_new]

wave_eq = tf.compile(WaveEq)
```

The tensor programs take and output tensor memory buffers, which can be created from numpy arrays:
```python
A = tf.tensor(np.zeros([100, 100], dtype=np.float32))
B = tf.tensor(np.zeros([100, 100], dtype=np.float32))
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
    N, M = A.shape
    B = tf.input([M, -1], tf.float32) #M must match
    K = B.shape[1]

    C = tf.zeros([N, K])
    i, j, k = tf.indices([N, K, M])
    tf.scatterAdd(C[i, j], A[i, k] * B[k, j])

    return [C]

matmul = tf.compile(MatrixMultiplication)
```

Here the 3D nature of the matrix multiplication is apparent. The scatter operation is used to accumulate the results of the row-column dot products into the elements of the resulting matrix.

(This is not the most efficient way to implement matrix multiplication, but it is the simplest way to show how scatter operations work. In the future though, some dimensions will be converted into loop indices, and the scatter operation will be used to accumulate the results of the dot products into the resulting matrix.)

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
- [ ] Compile from Python AST instead of tracing
  
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