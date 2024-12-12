# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install build twine

# Configure CMake
$env:PYTHON_EXECUTABLE = (Get-Command python).Path
$env:PYTHON_VERSION = (python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $env:PYTHON_VERSION"
echo "Python executable: $env:PYTHON_EXECUTABLE"

cmake -DPYBIND11_PYTHON_VERSION="$env:PYTHON_VERSION" `
      -DPython3_ROOT_DIR="$env:PYTHONLOCATION" `
      -DPython3_EXECUTABLE="$env:PYTHON_EXECUTABLE" `
      -DPython_FIND_STRATEGY=LOCATION `
      -DPython_FIND_REGISTRY=NEVER `
      -DPython_FIND_FRAMEWORK=NEVER `
      -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build the project with parallel jobs
$env:NUMBER_OF_PROCESSORS = (Get-WmiObject -Class Win32_ComputerSystem).NumberOfLogicalProcessors
cmake --build build --config Release --parallel $env:NUMBER_OF_PROCESSORS

# Build the wheel
python -m build -w ./Python