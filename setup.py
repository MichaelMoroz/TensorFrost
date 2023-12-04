from setuptools import setup

setup(
    name='TensorFrost',
    version='0.1',
    packages=['TensorFrost'],
    package_dir={'TensorFrost': 'python_module'},
    package_data={'TensorFrost': ['*.pyd']},
)