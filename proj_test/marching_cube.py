import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from skimage import measure

iso_treshold_0 = 0.1
iso_treshold_1 = 0.2
iso_treshold_2 = 0.4
iso_treshold_3 = 0.6

# Import all the frames data from results folder
files = glob.glob('results/frames_data/color_field_[0-9]*.npy')
# max_frame = len(files)
max_frame = 24*5

results = [np.load(file) for file in files[:max_frame]]
print(results[0])
# import mesh
mesh = np.load('results/frames_data/color_field_mesh.npy')

def plot_frame(frame_number, result_matrix):
    print("Frame number:", frame_number)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 20)  
    ax.set_ylim(0, 20)  
    ax.set_zlim(0, 20)
    
    try:

        # verts_0, faces_0, normals_0, values_0 = measure.marching_cubes(result_matrix[frame_number], iso_treshold_0)
        verts_1, faces_1, normals, values = measure.marching_cubes(result_matrix[frame_number], iso_treshold_1)
        verts_2, faces_2, normals, values = measure.marching_cubes(result_matrix[frame_number], iso_treshold_2)
        verts_3, faces_3, normals, values = measure.marching_cubes(result_matrix[frame_number], iso_treshold_3)

        # Plot the isosurfaces
        # mesh_0 = Poly3DCollection(verts_0[faces_0])
        mesh_1 = Poly3DCollection(verts_1[faces_1])
        mesh_2 = Poly3DCollection(verts_2[faces_2])
        mesh_3 = Poly3DCollection(verts_3[faces_3])

        # mesh_0.set_edgecolor((0, 0, 0.2, 0.1))
        mesh_1.set_edgecolor((0, 0, 0.2, 0.2))
        mesh_2.set_edgecolor((0, 0, 0.2, 0.4))
        mesh_3.set_edgecolor((0, 0, 0.2, 0.6))

        # mesh_0.set_facecolor((0, 0, 0.2, 0.1))
        mesh_1.set_facecolor((0, 0, 0.2, 0.2))
        mesh_2.set_facecolor((0, 0, 0.2, 0.4))
        mesh_3.set_facecolor((0, 0, 0.2, 0.6))

        # ax.add_collection3d(mesh_0)
        ax.add_collection3d(mesh_1)
        ax.add_collection3d(mesh_2)
        ax.add_collection3d(mesh_3)
        fig.savefig('results/marching_cubes/' + str(frame_number) + '.png')
        # close figure to avoid memory leak
        plt.close(fig)
    except:
        print("Error in frame", frame_number)


for i in range(max_frame):
    plot_frame(i, results)