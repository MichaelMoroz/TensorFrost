#!/bin/bash

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Define the list of Python versions
PYTHON_VERSIONS="3.7 3.8 3.9 3.10 3.11 3.12"

# Function to check if a specific Python version is installed
check_python_version() {
    if ! pyenv versions | grep -q "$1"; then
        echo "Python $1 not found, attempting to install..."
        pyenv install "$1"
    fi
}

clean_rebuild() {
    # Set the Python version for pyenv
    pyenv local "$1"
    
    # Reload the shell environment
    eval "$(pyenv init -)"

    # Refresh the shell environment to ensure pyenv shims are used
    export PATH="$(pyenv root)/shims:$PATH"

    # Get the path of the current Python interpreter
    PYTHON_PATH=$(pyenv which python)

    # Rest of your build commands
    git clean -d -x -f -e Python/dist
    cmake -DPYBIND11_PYTHON_VERSION="$1" -DPYTHON_EXECUTABLE="$PYTHON_PATH" -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    "$PYTHON_PATH" -m pip install build
    "$PYTHON_PATH" -m build -w ./Python
}

# Main loop
for VERSION in $PYTHON_VERSIONS; do
    echo "Building for Python $VERSION"
    check_python_version "$VERSION"
    clean_rebuild "$VERSION"
done
