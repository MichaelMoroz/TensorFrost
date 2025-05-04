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

$PYTHON_EXECUTABLE -m pip wheel ./Python -w Python/dist --verbose

for whl in Python/dist/*.whl; do
  auditwheel repair "$whl" -w ./wheelhouse/
done