from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None

with open("README.md", 'r') as f:
    long_description = f.read()


setup(
    name="Pyllab",
    version="0.0.2",
    packages=find_packages(),
    author="Riccardo Viviano, Matteo Galdi",
    description="Deep learning library",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/ez-spark/Pyllab",
    ext_modules = cythonize([Extension("llab", ["Pyllab/__init__.pyx"],libraries=["llab"])]),
)

