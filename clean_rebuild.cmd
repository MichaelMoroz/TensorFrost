git clean -d -x -f -e PythonBuild/dist
cmake -DPYBIND11_PYTHON_VERSION=3.12 -S . -B build 
cmake --build build
py -3.12 -m pip install build
py -3.12 -m build ./PythonBuild