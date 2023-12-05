git clean -d -f -x
cmake -DPYTHON_VERSION=3.12 -S . -B build 
cmake --build build
py -3.12 -m build ./PythonBuild