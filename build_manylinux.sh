#!/bin/bash
set -e
cd /io

# Install necessary dependencies
yum install -y epel-release kernel-headers wayland-devel libxkbcommon-devel \
  wayland-protocols-devel mesa-libGL-devel mesa-libEGL-devel libXcursor-devel \
  libXi-devel libXinerama-devel libXrandr-devel libXrender-devel libXext-devel \
  libXfixes-devel libXt-devel libXtst-devel libX11-devel libXdamage-devel \
  libXcomposite-devel libwayland-client libevdev-devel kernel-devel libXrandr-devel \
  && yum clean all

# Check if PYTHON_VERSION is set
if [ -z "$PYTHON_VERSION" ]; then
    echo "Error: PYTHON_VERSION environment variable is not set"
    exit 1
fi

export PYTHON_ROOT=/opt/python/$PYTHON_VERSION
export PATH=$PYTHON_ROOT/bin:$PATH
export PYTHON_EXECUTABLE=$PYTHON_ROOT/bin/python

#$PYTHON_EXECUTABLE -m pip install --upgrade pip -r requirements.txt cmake build auditwheel
#
## Extract Python version (e.g., 3.7, 3.8, etc.)
#PY_VERSION=$($PYTHON_EXECUTABLE -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
#
#echo "Python version: $PY_VERSION"
#echo "Python executable: $PYTHON_EXECUTABLE"
#
#$PYTHON_EXECUTABLE -m cmake \
#  -DPython3_ROOT_DIR=$PYTHON_ROOT \
#  -DPython3_EXECUTABLE=$PYTHON_EXECUTABLE \
#  -DPython_FIND_STRATEGY=LOCATION \
#  -DPython_FIND_REGISTRY=NEVER \
#  -DPython_FIND_FRAMEWORK=NEVER \
#  -DPYBIND11_PYTHON_VERSION=$PY_VERSION \
#  -DCMAKE_C_FLAGS="-D_POSIX_C_SOURCE=200809L -Wno-deprecated-declarations" \
#  -DCMAKE_CXX_FLAGS="-D_POSIX_C_SOURCE=200809L -Wno-deprecated-declarations" \
#  -S . -B build -DCMAKE_BUILD_TYPE=Release

$PYTHON_EXECUTABLE -m pip wheel ./Python -w Python/dist --verbose

for whl in Python/dist/*.whl; do
  auditwheel repair "$whl" -w ./wheelhouse/
done