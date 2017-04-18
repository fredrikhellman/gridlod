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

These packages might be needed:
* `PyEVTK`
* `scikit-sparse`
* `numpy`
* `scipy`

## Getting started
You can find some examples on how to use it in the test directory
`test`. See e.g. `gridlod/test/test_pg.py` or
`gridlod/integrations/test_pgtransport.py`. Especially, I think the
example `test_1d()` in `test_pg.py` is illustrative. The example
is taken from Daniel Peterseim's paper, *Variational Multiscale
Stabilization and the Exponential Decay of Correctors*.

## Basic terminology
The code has been developed in stages, and the terminology is not
consistent. A few pointers follow.

* `world` means the unit hypercube in question.
* `patch` means a subdomain that is also a hypercube.

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

## File structure

The base consists of these files:
* `fem.py` contains code for the assembly of finite element matrices.
* `interp.py` contains code for interpolation operators.
* `world.py` contains the world class.
* `coef.py` contains code for defining a coefficient.
* `femsolver.py` contains code for solving a reference FEM problem.
* `linalg.py` contains code for solving linear systems.
* `util.py` contains code for manipulating grid indices.
* `lod.py` contains code for solving the LOD patch problems.
* `transport.py` contains code for computing fluxes.

The application in *Numerical homogenization of time-dependent
diffusion* (arXiv:1703.08857) is defined by the following files.
* `pg.py` contains code for the main algorithm in that paper.
* `ecworker.py` and `eccontroller.py` contains a hack that parallelizes computations over an ipyparallel cluster.

## Tests

The test and integration directories contain some tests that can be
illustrative. Use e.g. `nosetests` to run the tests.
