import glob
import numpy as np

import pyvista as pv
from matplotlib import cm
import os
from pyvista import examples

class Renderer:
    def __init__(self, result_dir: str, radius: float, start_frame=0, max_frame=-1, render_boundary=False, render_density=False, render_temperature=False, mass=-1, SHOW=True, framestep=1, framerate=24, render_uniform=False, resolution = [1280,720]):

        self.result_dir = result_dir
        os.makedirs(self.result_dir + "frames", exist_ok=True)
        self.radius = radius
        self.framerate = framerate
        self.SHOW = SHOW
        self.render_boundary = render_boundary
        self.render_density = render_density
        self.render_temperature = render_temperature
        self.render_uniform = render_uniform
        self.mass = mass
        self.framestep = framestep
        self.start_frame = start_frame
        self.max_frame = max_frame
        self.resolution = resolution

         # All the files from results folder
        if max_frame == -1:
            files_counter = glob.glob(result_dir+'data/frame_[0-9]*.npy')
            self.max_frame = len(files_counter)
        else:
            self.max_frame = max_frame

        print(self.max_frame)
        # Ordered files:
        if render_uniform:
            self.files = [f"{result_dir}data/frame_uniform_{i}.npy" for i in range(self.max_frame)]
            self.uniform_pos = np.load(result_dir+'data/uniform_pos.npy')
        else:
            self.files = [f"{result_dir}data/frame_{i}.npy" for i in range(self.max_frame)]

        if self.render_density:
            self.mass = mass
            self.density_files = [f"{result_dir}data/frame_density_{i}.npy" for i in range(self.max_frame)]
            self.cmap = cm.jet
        elif self.render_temperature:
            self.temperature_files = [f"{result_dir}data/frame_temperature_{i}.npy" for i in range(self.max_frame)]
            self.cmap = cm.jet
            init_data = np.load(self.temperature_files[0])
            self.clim = [15, 300]

        if render_boundary:
            self.b_particles_data = np.load(result_dir+'data/boundary.npy')

    def render(self):
        if self.render_uniform:
            self.render_uniform_grid()
        else:
            self.render_particles()

    def render_uniform_grid(self):
        cubemap = examples.download_sky_box_cube_map()

        for frame_id in range(self.start_frame, self.max_frame, self.framestep):
            p = pv.Plotter(off_screen=True)
            p.add_actor(cubemap.to_skybox())
            p.set_environment_texture(cubemap)
            print(frame_id, end="\r")

            field_data = np.load(self.files[frame_id])
            grid_size = field_data.shape[0]
            new_field_data = np.empty([grid_size*field_data.shape[-1]]*3)
            it = np.nditer(field_data, flags=['multi_index'])
            for x in it:
                # x, it.index
                index = it.multi_index
                new_field_data[
                    index[0]*4+ index[3],
                    index[1]*4+ index[4],
                    index[2]*4+ index[5]
                ] = x
                
            # grid = pv.UniformGrid()
            # grid.dimensions = field_data.shape
            # grid.spacing = (1, 2, 1)
            # grid.origin = (0, 0, 0)
            # print(field_data)
            # mesh = grid.contour([250], field_data, method="marching_cubes")
            volume_data = pv.wrap(new_field_data)
            volume_data.origin = (0, 0, 0)
            volume_data.spacing = (1, 1, 1)
            surface = volume_data.contour([150])
            if len(surface.points) > 0:
                p.add_mesh(surface, color="blue", pbr=True, metallic=0.01, roughness=0.1, diffuse=1)
            # p.add_volume(volume_data, cmap="bone")
            if self.render_boundary:
                pass
                # mesh = pv.wrap(self.b_particles_data)
                # mesh.origin = (0, 0, 0)
                # p.add_mesh(mesh, point_size=3, render_points_as_spheres=True, opacity=0.05)


            # if self.render_boundary:
                # p.add_mesh(self.b_particles_data, point_size=3, render_points_as_spheres=True, opacity=0.01)
            # cpos = [(50.3838307134328, 130.20751731316425, 32.18513266960956),
            #         (0, 0, 0),
            #         (0.018, 0.99, -0.06)]
            cpos = [(200, 100, 200), (0.0, 0.0, 0.0), (0.018, 0.99, -0.06)]
            p.show(screenshot=self.result_dir+f"frames/{frame_id:06d}.png", cpos=cpos)
            # p.show(screenshot=self.result_dir+f"frames/{frame_id:06d}.png")

    def render_particles(self):   
        for frame_id in range(self.start_frame, self.max_frame, self.framestep):
            p = pv.Plotter(off_screen=True, window_size=self.resolution)
            print(frame_id, end="\r")
            particles = np.load(self.files[frame_id])
            if self.render_density:
                density = np.load(self.density_files[frame_id])
                p.add_mesh(particles, scalars=density, point_size=10, render_points_as_spheres=True, cmap=self.cmap, clim=[150, 300])
            elif self.render_temperature:
                temperature = np.load(self.temperature_files[frame_id])
                p.add_mesh(particles, scalars=temperature, point_size=10, render_points_as_spheres=True, cmap=self.cmap, clim=self.clim)
            else:
                p.add_mesh(particles, point_size=10, render_points_as_spheres=True)
            if self.render_boundary:
                p.add_mesh(self.b_particles_data, point_size=3, render_points_as_spheres=True, opacity=0.05)
            
            # print(particles.shape)
            cpos = [(10, 5, 10), (0.0, 0.0, 0.0), (0.018, 0.99, -0.06)]

            p.show(screenshot=self.result_dir+f"frames/{frame_id:06d}.png", cpos=cpos)


