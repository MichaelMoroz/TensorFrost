@echo off
set PYTHON_VERSION=%1
if "%PYTHON_VERSION%"=="" set PYTHON_VERSION=3.12

git clean -d -x -f -e PythonBuild/dist
cmake -DPYBIND11_PYTHON_VERSION=%PYTHON_VERSION% -S . -B build -D CMAKE_BUILD_TYPE=Release 
cmake --build build 
py -%PYTHON_VERSION% -m pip install build
py -%PYTHON_VERSION% -m build -w ./PythonBuild