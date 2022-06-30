from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
from Cython.Distutils import build_ext
import numpy
import os
import sysconfig

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None

def get_ext_filename_without_platform_suffix(filename):
    name, ext = os.path.splitext(filename)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    if ext_suffix == ext:
        return filename

    ext_suffix = ext_suffix.replace(ext, '')
    idx = name.find(ext_suffix)

    if idx == -1:
        return filename
    else:
        return name[:idx] + ext

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
        
    # deleting long name cpython.... for .so or .pyd file
    '''
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename('Pyllab')
        return get_ext_filename_without_platform_suffix(filename)
    '''
    
setup(
    name="Pyllab",
    version="1.0.0",
    cmdclass={'build_ext': Build},
    packages=find_packages(),
    setup_requires=['setuptools>=18.0','wheel','cython', 'numpy'],
    author="Riccardo Viviano, Matteo Galdi",
    author_email="riccardoviviano@gmail.com",
    description="Deep learning library",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/ez-spark/Pyllab",
    ext_modules = cythonize([Extension("pyllab",
                                       ["Pyllab/*.pyx"],
                                       include_dirs=['./src/', numpy.get_include()],
                                       libraries=["llab"],
                                       library_dirs = ['./'], 
                                       extra_link_args=["-DSOME_DEFINE_OPT", "-L . "],
                                       extra_compile_args=["-O3", "-mavx2"])],
                            compiler_directives={'language_level' : "3"})
)

