import pyvista as pv
import numpy as np
import os

def getFormat(file: str):
    # incredible one-liner
    return file[file.find('.') + 1:]

def readParticles_multi(filearray, align_modes = [], offsetarray = []):
    """currently only support for vtk\n
    align_mode 0(leave data as is)\n
    align_mode 1(align into origin corner)\n
    align_mode 2(center around origin)\n"""
    points = []
    i = 0
    if len(filearray) == 0:
        return np.array([[2.,0.2,2.], [2.1,0.2,2.]],float)
    for file in filearray:
        if len(points) == 0:
            points = readParticles(file, align_modes[i], offsetarray[i])
        else:
            points = np.append(points, readParticles(file, align_modes[i], offsetarray[i]), axis = 0)
        i+=1
    return points

def readParticles(file, align_mode = 0, offset = np.array([0.0, 0.0, 0.0])):
    """currently only support for vtk or txt (metafile)\n
    align_mode 0(leave data as is)\n
    align_mode 1(align into origin corner)\n
    align_mode 2(center around origin)\n"""
    format = getFormat(file)
    if format == "txt":
        filenames = []
        align_modes = []
        offsets = []
        with open(file) as f:
            while True:
                pointdata = f.readline()
                if pointdata == "":
                    break
                pointdata = os.path.join("src", "pointDataFiles", pointdata)
                filenames.append(pointdata.replace('\n',''))
                align_modes.append(int(f.readline()))
                offset_txt = str(f.readline().replace('\n',''))
                offset = np.fromstring(offset_txt, dtype = float, sep = ' ')
                offsets.append(offset)
                f.readline()
        # print(filenames)
        # print(align_modes)
        # print(offsets)
        points = readParticles_multi(filenames, align_modes, offsets)
        return points

    points = None
    if format == "vtk":
        mesh = pv.read(file)
        points = mesh.points
    elif format == "bgeo":
        # TODO, may not be needed
        pass
    elif format == "npy":
        points = np.load(file)
        # slice away zeroes (could be wrong)
        points = np.array(points[points != np.array([0., 0., 0.])]).reshape((-1,3))


    
    if align_mode == 1:
        # even better one-liner
        for coord in range(3): points[0:,coord] -= min(points[0:,coord])
    elif align_mode == 2:
        for coord in range(3): points[0:,coord] -= (min(points[0:,coord])+max(points[0:,coord]))/2
    elif align_mode == 3:
        points[0:,1] -= min(points[0:,1])
        for coord in (0,2): points[0:,coord] -= (min(points[0:,coord])+max(points[0:,coord]))/2

    points[:,] += offset
    return points    




def main(file):
    points = readParticles(file)
    print("shape of array:")
    print(points.shape)
    print(points)
    x_coords = points[0:,0]
    print("x_range: ")
    print((min(x_coords),max(x_coords)))

    y_coords = points[0:,1]
    print("y_range: ")
    print((min(y_coords),max(y_coords)))

    z_coords = points[0:,2]
    print("z_range: ")
    print((min(z_coords),max(z_coords)))

if __name__ == '__main__':
    main(r"src\pointDataFiles\erlenmayer_quickerstart.npy")