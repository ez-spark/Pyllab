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

- It is a pain in the Ass. Why?

Pyllab is a cython library compiling .C files that uses poix calls system. Now you can see the problem here. Just follow me in this journey:


- Install Mingw with MYSYS2: https://www.msys2.org/ follow the steps and also the passages to install mingw-w64 

-  now navigate to Pyllab with MYSYS2 and create the .dll library:

```
sh create_shared_windows.sh
```

- Go to your Python Folder (We assume as example we are using python 3.7):

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

- Now you can run from MYSYS2 sh build_python_library.sh
