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

## How to Compile

Instructions on `code/README.md`.
