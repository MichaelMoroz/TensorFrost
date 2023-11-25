# 🥶 TensorFrost 🥶
Yet another Python tensor library with autodifferentiation (TODO). Currently very much a work in progress.

The main idea of this library is to compile optimal fused kernels for the GPU given a set of numpy-ish functions, including other more complex things like loops and conditionals (also TODO)

Currently working platforms:
| Backend/OS | CPU | CUDA | Vulkan |
|------------|-----|------|--------|
| Windows    | 🚧   | ⛔    | ⛔      |
| Linux      | ⛔   | ⛔    | ⛔      |

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

Then you can create and compile functions:
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

TODO: advanded usage and debugging

## Contributing
Contributions are welcome! If you want to contribute, please open an issue first to discuss the changes you want to make.