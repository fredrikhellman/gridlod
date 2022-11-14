# gridlod

`gridlod` is a python code for computing localized element correctors
used in the localized orthogonal decomposition (LOD) method for
numerical homogenization. The code works for any number of dimensions
*d* (also greater than 3) but is limited to computations on the
hypercube *[0,1]^d* on a structured grid as coarse mesh, and another
structured grid (refinement of the former) as fine mesh.

The code is written with consideration of the memory usage. Thus, the
Petrov--Galerkin formulation is used. Also, fine-scale quantities and
coarse-scale quantities are distinguished to be able to discard as
much fine-scale information as early as possible to reduce the memory
usage.

## Required packages

These `python` packages are needed:
* `scikit-sparse`
* `numpy`
* `scipy`

For installing [`scikit-sparse`](https://github.com/scikit-sparse/scikit-sparse),
the library `suite-sparse` is required.
As also highlighted [here](https://github.com/scikit-sparse/scikit-sparse),
there may appear problems installing `scikit-sparse` on a Mac,
which is due to the `suite-sparse` location not being installed at the right place.
For fixing this, just find out your `lib` and `include` directory of your `suite-sparse`
and use the respective `pip install` command which defines
`SUITESPARSE_INCLUDE_DIR` and `SUITESPARSE_LIBRARY_DIR` manually.

## Getting started
You can find some examples on how to use it in the test directory
`test`. See e.g. the example in `gridlod/test/test_pgexamples.py`. The
example is taken from Daniel Peterseim's paper, *Variational
Multiscale Stabilization and the Exponential Decay of Correctors*.

## Basic terminology
The code has been developed in stages, and the terminology is not
consistent. A few pointers follow.

* `world` means the unit hypercube in question.
* `patch` means a subdomain that is also a hypercube but where one
  particular element is special (typically the center element of the
  patch for corrector problems).

Hypercubes are described by arrays. E.g. `NPatch = np.array([5,2,6])`
is a 5 times 2 time 6 element hypercube.

To allow for two meshes, we talk about Coarse and Fine
quantities. `NWorldCoarse` for example is the extent of the world in
coarse elements, and `NWorldFine` the extent in fine elements. The
elementwise ratio `NWorldFine/NWorldCoarse` is consistently called
`NCoarseElement` and should be all integers.

All domains are hypercubes and can thus generally be described by its
extent (typically `NPatchCoarse`) and a starting index, cartesian
(`iPatchCoarse`) or lexicographical (with varying names).

## Linear storage of data
When data (e.g. elementwise coefficient or nodewise solution vector)
is stored linearly, it is stored with the first index running
fastest. E.g.: `aFine` is typically the coefficient. `aFine[0]` is
coefficient at fine index *(0, 0, 0)*, `aFine[1]` at *(1, 0 0)*, and
so on.

## Index arithmetic
There is a lot of index arithmetic going on to be able to do bulk
operations with the numpy routines. The `util.py` file contains
several routines for constructing subpatches or finding a
lexicographical index from a cartesian coordinate and so on. The basis
of many routines is `pIndexMap`.

## Files

The base consists of these files:
* `fem.py` contains code for the assembly of finite element matrices.
* `interp.py` contains code for interpolation operators.
* `world.py` contains the World and Patch class.
* `coef.py` contains code for localizing a coefficient.
* `femsolver.py` contains code for solving a reference FEM problem.
* `linalg.py` contains code for solving linear systems.
* `util.py` contains code for manipulating grid indices.
* `lod.py` contains code for solving corrector problems.
* `pglod.py` contains code for assembling PG-LOD matrices
* `func.py` contains code for describing Q0 and Q1 functions.
* `transport.py` contains code for computing fluxes (not well-maintained)

## Tests

The test and integration directories contain some tests that can be
illustrative. Use e.g. `nosetests` to run the tests.

## Contributors

* Fredrik Hellman
* Tim Keil