from setuptools import setup

#load readme
with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='TensorFrost',
    version='0.3.1',
    author="Mykhailo Moroz",
    author_email="michael08840884@gmail.com",
    description="Tensor library with automatic kernel fusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MichaelMoroz/TensorFrost",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Environment :: Win32 (MS Windows)",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.7',
    packages=["TensorFrost"],

    # Include pre-compiled extension
    package_data={"TensorFrost": ["*.so", "*.pyd"]},
    has_ext_modules=lambda: True
)
