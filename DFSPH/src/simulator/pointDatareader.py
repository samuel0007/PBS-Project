import pyvista as pv
import numpy as np

# This may not be needed, since it's all vtk for now
def getFormat(file: str):
    return file[file.find('.') + 1:]

def readParticles(file, align_mode = 0):
    """currently only support for vtk
    align_mode 0(leave data as is)
    align_mode 1(align into origin corner)
    align_mode 2(center around origin)"""
    format = getFormat(file)
    points = None
    if format == "vtk":
        mesh = pv.read(file)
        points = mesh.points
    elif format == "bgeo":
        # TODO, may not be needed
        pass
    
    if align_mode == 1:
        for coord in range(3): points[0:,coord] -= min(points[0:,coord])
    elif align_mode == 2:
        for coord in range(3): points[0:,coord] -= (min(points[0:,coord])+max(points[0:,coord]))/2

    return points    


def readParticles_multi(filearray, align_mode = 0):
    """currently only support for vtk
    align_mode 0(leave data as is)
    align_mode 1(align into origin corner)
    align_mode 2(center around origin)"""
    points = []
    for file in filearray:
        if len(points) == 0:
            points = readParticles(file, align_mode)
        else:
            points = np.append(points, readParticles(file, align_mode), axis = 0)
    return points

def main(file):
    points = readParticles_multi([file,file])
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