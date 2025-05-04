@echo off
set PYTHON_VERSION=%1
if "%PYTHON_VERSION%"=="" set PYTHON_VERSION=3.12

git clean -d -x -f -e Python/dist
py -%PYTHON_VERSION% -m pip install -e Python/ -v