import pyvista as pv
mesh = pv.read("bunny.vtk")
print(mesh.points)

pl = pv.Plotter()
pl.add_points(mesh.points, color="red", point_size=50, render_points_as_spheres=True)
pl.show()