from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext

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

link_args = ['-static-libgcc',
             '-static-libstdc++',
             '-Wl,-Bstatic,--whole-archive',
             '-lwinpthread',
             '-Wl,--no-whole-archive',
             '-L .']

class Build(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == 'mingw32':
            for e in self.extensions:
                e.extra_link_args = link_args
        super(Build, self).build_extensions()

setup(
    name="Pyllab",
    version="0.0.2",
    cmdclass={'build_ext': Build},
    packages=find_packages(),
    setup_requires=['wheel'],
    author="Riccardo Viviano, Matteo Galdi",
    description="Deep learning library",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/ez-spark/Pyllab",
    ext_modules = cythonize([Extension("llab", ["Pyllab/__init__.pyx"],libraries=["llab"],extra_link_args=["-DSOME_DEFINE_OPT", "-L . "],extra_compile_args=["-O3","-mavx2"])]),
    

)

