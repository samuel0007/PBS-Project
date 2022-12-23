# References

These are the references we looked into during our research, though not all of them were used in the end.

## fluid-fluid interaction

Base paper. 

## kernels

Contains the different kernel functions.

## Position Based Fluids

Introduce density constraints to avoid colloidal instability.

## Cool website with a lot of informations on SPH

https://interactivecomputergraphics.github.io/physics-simulation/

## Survey on current SPH methods

https://onlinelibrary.wiley.com/doi/10.1111/cgf.14508


## SPH for fluids tutorial

Quite extensive paper in which they describe the full implementation of a modern SPH solver.


## SPH for viscous fluid

https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.13349

## SPH for multiphase fluids

https://www.researchgate.net/publication/220789300_Density_Contrast_SPH_Interfaces

## SPH drag force

Might be usefull looking at applying some drag due to air friction when the fluid exits the becher.

https://www.sciencedirect.com/science/article/pii/S0097849317301541

## MLS_localPressureBoundaries

Boundary method that claims to be better than Akinci et al.

## SPH multiple fluids using mixture model

Multiple-Fluid SPH Simulation Using a Mixture Model

## Opensource Library for SPH based on Eigen, implements nearly all the methods described in the papers above

https://splishsplash.readthedocs.io/en/latest/about.html

We also got the dragon model from splishsplash

## Tool for Surface/Volume sampling

Surface sampling used for boundary conditions.

Volume sampling used for initial conditions.

https://splishsplash.readthedocs.io/en/latest/SurfaceSampling.html\
https://splishsplash.readthedocs.io/en/latest/VolumeSampling.html

(For surface sampling, we modified the source code by copying stuff \
from the VolumsSampling folder such that it can output vtk files)

# SplishSplash Interesting files

## SPHKernels.h

Contains common SPH kernels as well as their gradients.

## Flask model

https://www.turbosquid.com/3D-Models/free-glass-erlenmayer-flask-3d-model/433384

