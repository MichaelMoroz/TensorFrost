from setuptools import setup

setup(
    name='test_pybind',
    version='1.0',
    packages=[''],
    package_dir={'': 'python_module'},
    package_data={'': ['*.pyd']},
)