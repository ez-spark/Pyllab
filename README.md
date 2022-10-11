# Install
```
pip install Pyllab
```

Warning:

from pip you will download the corresponding .whl file according to your OS and python version.
All the .whl files have been created from the compilation of C code. The files on pypi have been compiled
for the x86_64 architecture with the extensions "-mavx2". So, if you have a pentium for example
or a processor that does not support the extension avx2 you can find in the github repo in the releases
the different .whl files with: no extensions, sse extension, sse2 extension, avx extension.

or


# Build .whl package On Linux:
```
pip install -r requirements.txt
sh generate_wheel_unix.sh
```

- On Linux specifically you have to fix the wheel package, use the container with the repair tool:

```
sudo docker run -i -t -v `pwd`:/io quay.io/pypa/manylinux1_x86_64 /bin/bash
```

when you are in run these lines:

```
cd io
sh repair_wheel_linux.sh
```

in the wheelhouse directory you have the fixed wheel package

# Build .whl package on MacOS
```
pip install -r requirements.txt
sh generate_wheel_unix.sh
sh repair_wheel_macos.sh
```

# Build .whl package on Windows

- It is a PIA. Why?

Pyllab is a cython library compiling .C files that use posix calls system. Now you can see the problem here. Just follow me in this journey:

- First of all you need to comment this line in __init__.pyx:

```
ctypedef stdint.uint64_t uint64_t
```

- Install Mingw with MYSYS2: https://www.msys2.org/ follow the steps and also the passages to install mingw-w64 

-  now navigate to Pyllab with MYSYS2 and create the .lib library:

```
sh create_library.sh
```

- Go to your Python Folder (We assume as example we are using python 3.7):

- You can find your python folder with:

```
import sys

locate_python = sys.exec_prefix
print(locate_python)
```

- Create the file Python37\Lib\distutils\distutils.cfg that should look like this:

```
[build]
compiler=mingw32
 
[build_ext]
compiler=mingw32
```

- Download this .dll file https://it.dllfile.net/microsoft/vcruntime140-dll:

- Move it to Python37\libs and to Python37\DLLs

- Move libllab.lib previously built to Python37\libs

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

 ```
 /* Compiler specific defines */
 

#ifdef __MINGW32__
#ifdef _WIN64
#define MS_WIN64
#endif
#endif
```

- Now you can run from MYSYS2:

```
sh build_python_library.sh
```

- Links:
 - https://www.msys2.org/ (To get MYSYS2 with mingw-w64)
 - https://wiki.python.org/moin/WindowsCompilers (distutils.cfg)
 - https://datatofish.com/locate-python-windows/ (Python folder search script)
 - https://stackoverflow.com/questions/34135280/valueerror-unknown-ms-compiler-version-1900 (vcruntime140-dll lacking dll)
 - https://bugs.python.org/file40608/patch.diff (function fix, changing from 1900 to 1916 was my guess and it worked, it enables the use of the .dll previously installed)
 - https://github.com/cython/cython/issues/3405 (For the ifdef stuff, cython bug)

# Install .whl files

Once you have created the .whl file, you can install it locally using pip:

```
pip install package.whl
```
# Import the library in python

```
import pyllab
```

# Pyllab supports

- Version: 1.0.0

 - [x] Fully connected Layers
 - [x] Convolutional Layers
 - [x] Transposed Convolutional Layers
 - [x] Residual Layers
 - [x] Dropout
 - [x] Layer normalization for Fully-connected Layers Transposed Convolutional and Convolutional Layers
 - [x] Group normalization for Convolutional and Transposed Convolutional Layers
 - [x] 2d Max Pooling
 - [x] 2d Avarage Pooling
 - [x] 2d Padding
 - [x] Local Response Normalization for Fully-connected, Convolutional, Transposed Convolutional Layers
 - [x] sigmoid function
 - [x] relu function
 - [x] softmax function
 - [x] leaky_relu function
 - [x] elu function
 - [x] standard gd and sgd
 - [x] Nesterov optimization algorithm
 - [x] ADAM optimization algorithm
 - [x] RADAM optimization algorithm
 - [x] DiffGrad optimization algorithm
 - [x] ADAMOD optimization algorithm
 - [x] Cross Entropy Loss
 - [x] Focal Loss
 - [x] Huber Loss type1
 - [x] Huber Loss type2
 - [x] MSE Loss
 - [x] KL Divergence Loss
 - [x] Entropy Loss
 - [x] Total Variational Loss
 - [x] Contrastive 2D Loss
 - [x] Edge Pop-up algorithm
 - [x] Dueling Categorical DQN
 - [x] Rainbow Training
 - [x] Genetic Algorithm training (NEAT)
 - [x] Multi Thread
 - [x] Numpy input arrays
 - [ ] GPU Training and inference (Future implementation)
 - [ ] RNN
 - [ ] LSTM (Future implementation already tested in C)
 - [ ] Transformers (Future implementation semi-implemented in C)
 - [ ] Attention mechanism (Future implementation already tested in C)
 - [ ] Multi-head Attention mechanism (Future implementation already tested in C)

# Genome API

```
import pyllab
# Init a genome from a .bin file
g = pyllab.Genome("file.bin", input_size, output_size)
# Get the output from an input list
inputs = [1]*input_size
output = g.ff(inputs)
```

# DL Model API

```
import pyllab
# Init a model from a .txt file
model = pyllab.Model(pyllab.get_dict_from_model_setup_file("./model/model_023.txt"))
# select the training mode (default is standard training, otherwise you can choose edge oppup)
percentage_of_used_weights_per_layer = 0.5
model.set_training_edge_popup(percentage_of_used_weights_per_layer)
# select the multi-thread option
model.make_multi_thread(batch_size)
# select the loss function
model.set_model_error(pyllab.PY_FOCAL_LOSS,model.get_output_dimension_from_model(),gamma=2)
# init the optimization hyperparameters
train = pyllab.Training(lr = 0.01, momentum = 0.9,batch_size = batch_size,gradient_descent_flag = pyllab.PY_ADAM,current_beta1 = pyllab.PY_BETA1_ADAM,current_beta2 = pyllab.PY_BETA2_ADAM, regularization = pyllab.PY_NO_REGULARIZATION,total_number_weights = 0, lambda_value = 0, lr_decay_flag = pyllab.PY_LR_NO_DECAY,timestep_threshold = 0,lr_minimum = 0,lr_maximum = 1,decay = 0)
# train in supervised mode on a bunch of data
for i in range(epochs):
    # save the model in a binary file "i.bin"
    model.save(i)
    inputs, outputs = shuffle(inputs, outputs)
    for j in range(0,inputs.shape[0],batch_size):
        # compute feedforward, error, backpropagation
        model.ff_error_bp_opt_multi_thread(1, 28,28, inputs[j:j+batch_size], outputs[j:j+batch_size], model.get_output_dimension_from_model())
        # sum the partial derivatives over the batch
        model.sum_models_partial_derivatives()
        # update the model according to the optimization hyperparameters
        train.update_model(model)
        # update the optimization hyperparameters
        train.update_parameters()
        # reset the needed structures for another iteration
        model.reset()
```

# Rainbow API

Look at the rainbow.py file in the test directory

