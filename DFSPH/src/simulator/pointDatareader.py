import pyvista as pv
import taichi as ti

@ti.func
def readParticles(file):
    mesh = pv.read(file)
    return mesh.points