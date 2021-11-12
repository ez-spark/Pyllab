# Install
```
 pip install Pyllab==0.0.1
```

or


# Build .whl package On Linux/Mac:
```
sh create_dynamic.sh
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
