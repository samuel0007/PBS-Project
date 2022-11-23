import pyvista as pv
import taichi as ti


def readParticles(file):
    mesh = pv.read(file)
    return mesh.points

def main(file):
    mesh = pv.read(file)
    points = mesh.points
    print("shape of array:")
    print(points.shape)
    print(points)
    x_coords = points[0:-1,0]
    print("x_range: ")
    print((min(x_coords),max(x_coords)))

    y_coords = points[0:-1,1]
    print("y_range: ")
    print((min(y_coords),max(y_coords)))

    z_coords = points[0:-1,2]
    print("z_range: ")
    print((min(z_coords),max(z_coords)))

if __name__ == '__main__':
    main(r"src\pointDataFiles\Dragon.vtk")