import pyvista as pv

def getFormat(file: str):
    return file[file.find('.') + 1:]

def readParticles(file):
    format = getFormat(file)
    if format == "vtk":
        mesh = pv.read(file)
        return mesh.points
    if format == "bgeo":
        # TODO
        pass

def main(file):
    print(getFormat(file))
    points = readParticles(file)
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
    main(r"src\pointDataFiles\BigDragonSurface.vtk")