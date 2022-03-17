# Tests:

	- C Tests:
	
		+ C tests are used to check memory leaks
	
	- Python Tests:
	
		+ Python Tests are used to see if the networks learn in python

# Setup

- unpack the .tar package
- compile the .C files
- You need valgrind

```
make
```

# Run

```
sh run_memchecks.sh
```

runs the memcheck.c files that use all kind of functions with different models stored in the directory model

```
./dqn
```

Executes all the basic functions used by dqn.c

```
./neat
```

Executes all the basic functions used by neat.c

```
python dqn.py
```

A learning agent tries to learn cart-pole game from gym

```
rainbow.py
```

A learning agent tries to learn cart-pole game from gym

