# Install
```
 pip install Pyllab
```

or


# Build .whl package On Linux/Mac:
```
sh create_shared.sh
sh build_python_library.sh
```

- On Linux specifically run:

```
docker run -i -t -v `pwd`:/io quay.io/pypa/manylinux1_x86_64 /bin/bash
```

and

```
auditwheel repair package.whl
```

To repair the .whl file

# Build .whl package on Windows

- It is a PIA. Why?

Pyllab is a cython library compiling .C files that use posix calls system. Now you can see the problem here. Just follow me in this journey:


- Install Mingw with MYSYS2: https://www.msys2.org/ follow the steps and also the passages to install mingw-w64 

-  now navigate to Pyllab with MYSYS2 and create the .dll library:

```
sh create_shared_windows.sh
```

- Go to your Python Folder (We assume as example we are using python 3.7):

- You can find your python folder with:

```
import sys

locate_python = sys.exec_prefix
print(locate_python)
```

- Create the file Python37\Lib\distutils\distutils.cfg that should look like this:

[build]
compiler=mingw32
 
[build_ext]
compiler=mingw32

- Download this .dll file https://it.dllfile.net/microsoft/vcruntime140-dll:

- Move it to Python37\libs and to Python37\DLLs

- Move libllab.dll previously built to Python37\libs

- Now, Move to Python37\Lib\distutils\cygwinccompiler.py and modify the function get_msvcr() as follows: https://bugs.python.org/file40608/patch.diff

- The last part

```
elif msc_ver == '1900':
```

change it to 

```
elif msc_ver == '1916':
```

- Now other bug fix: go to Python37\include\pyconfig.h and add these lines:

 /* Compiler specific defines */
 
 ```
#ifdef __MINGW32__
#ifdef _WIN64
#define MS_WIN64
#endif
#endif
```

- Now you can run from MYSYS2 sh build_python_library.sh

- Links:
 - https://www.msys2.org/ (To got MYSYS2 with mingw-w64)
 - https://wiki.python.org/moin/WindowsCompilers (distutils.cfg)
 - https://datatofish.com/locate-python-windows/ (Python folder search script)
 - https://stackoverflow.com/questions/34135280/valueerror-unknown-ms-compiler-version-1900 (vcruntime140-dll lacking dll)
 - https://bugs.python.org/file40608/patch.diff (function fix, changing from 1900 to 1916 was my guess and it worked, it enables the use of the .dll previously installed)
 - https://github.com/cython/cython/issues/3405 (For the ifdef stuff, cython bug)
