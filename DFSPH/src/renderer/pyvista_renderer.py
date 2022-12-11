import glob
import numpy as np

import pyvista as pv
from matplotlib import cm
import os

class Renderer:
    def __init__(self, result_dir: str, radius: float, start_frame=0, max_frame=-1, render_boundary=False, render_density=False, render_temperature=False, mass=-1, SHOW=True, framestep=1, framerate=24, render_uniform=False):

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
            self.clim = [np.min(init_data), np.max(init_data)]

        if render_boundary:
            self.b_particles_data = np.load(result_dir+'data/boundary.npy')

    def render(self):
        if self.render_uniform:
            self.render_uniform_grid()
        else:
            self.render_particles()

    def render_uniform_grid(self):
        for frame_id in range(self.start_frame, self.max_frame, self.framestep):
            p = pv.Plotter(off_screen=True)
            print(frame_id, end="\r")

            field_data = np.load(self.files[frame_id])
            # grid = pv.UniformGrid()
            # grid.dimensions = field_data.shape
            # grid.spacing = (1, 2, 1)
            # grid.origin = (0, 0, 0)
            # print(field_data)
            # mesh = grid.contour([250], field_data, method="marching_cubes")
            volume_data = pv.wrap(field_data)
            volume_data.origin = (0, 0, 0)
            volume_data.spacing = (1, 2, 1)
            surface = volume_data.contour([80, 100, 150, 200, 250, 300])
            if len(surface.points) > 0:
                p.add_mesh(surface, color="blue", opacity=0.8)
            # p.add_volume(volume_data)

            # if self.render_boundary:
                # p.add_mesh(self.b_particles_data, point_size=3, render_points_as_spheres=True, opacity=0.01)
            cpos = [(50.3838307134328, 130.20751731316425, 32.18513266960956),
                    (0, 0, 0),
                    (0.018, 0.99, -0.06)]
            # cpos = [(170.63279823814065, 205.13279823814065, 160.63279823814065), (0.0, 0.0, 0.0), (0.018, 0.99, -0.06)]
            p.show(screenshot=self.result_dir+f"frames/{frame_id:06d}.png", cpos=cpos)
            # p.show(screenshot=self.result_dir+f"frames/{frame_id:06d}.png")

    def render_particles(self):   
        for frame_id in range(self.start_frame, self.max_frame, self.framestep):
            p = pv.Plotter(off_screen=True)
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
                p.add_mesh(self.b_particles_data, point_size=3, render_points_as_spheres=True, opacity=0.01)
            
            # print(particles.shape)
            cpos = [(10, 5, 10), (0.0, 0.0, 0.0), (0.018, 0.99, -0.06)]

            p.show(screenshot=self.result_dir+f"frames/{frame_id:06d}.png", cpos=cpos)


