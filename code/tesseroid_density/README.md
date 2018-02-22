# Variable density tesseroid gravity field computation

Here we can find the source code for the gravity fields computation.

## Files:

* `tesseroid.py`:
    Definition of the Python functions used by the user for the forward model
    computation. It calls the Cython precompiled functions.

* `tesseroid.pyx`:
    Definition of functions to compute the gravity fields written in Cython.
    They are no intended to be used by the user. Call the `tesseroid.py`
    functions instead.

* `Makefile`:
    File to automate the compilation of the `tesseroid.pyx` code.

## How to Compile

Run the following command:

```
make
```

## Requirements

You'll need a C compiler installed in order to build the Cython module.
On Linux, gcc should be installed by default.
On Windows, you'll need to install the Microsoft compiler for Python 2.7
(download at http://aka.ms/vcpython27).
