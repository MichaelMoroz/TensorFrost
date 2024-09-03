# Use the manylinux2014 image as a base
FROM quay.io/pypa/manylinux2014_x86_64

# Set environment variables
ENV PLAT manylinux2014_x86_64

# Copy the project files into the container
COPY . /io

# Set the working directory
WORKDIR /io

# Make the build script executable
RUN chmod +x /io/build_manylinux.sh

# Run the build script
CMD ["/bin/bash", "/io/build_manylinux.sh"]