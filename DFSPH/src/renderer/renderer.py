import taichi as ti
import numpy as np
import glob
import math


CAMERA_LOOK_AT = [0.5, 0.1, 0.5]
#TODO: draw boundary particles according to their pseudomass at each frame
REST_DENSITY = 1000

@ti.kernel
def compute_colors_from_density(density: ti.template(), colors: ti.template()):
    max_density_error = 0.

    for i in range(density.shape[0]):
        max_density_error = max(max_density_error, abs(density[i] - REST_DENSITY))

    for i in range(colors.shape[0]):
        value = abs(density[i] - REST_DENSITY) / max_density_error
        colors[i] = ti.Vector([value*0.4, value*0.6, value*0.8], ti.f32 )

class Renderer:
    def __init__(self, bounds: ti.f32, result_dir: str, radius: ti.f32, start_frame=0, max_frame=-1, render_boundary=False, render_density=False, mass=-1, SHOW=True, framerate=24):
        self.bounds = bounds
        self.x_bound = bounds
        self.y_bound = bounds
        self.z_bound = bounds

        # self.CAMERA_POS = [1.2*self.x_bound, 1.2*self.y_bound, 1.2*self.z_bound]
        self.CAMERA_POS = [1.4, 1.4, 2.4]
        self.render_boundary = render_boundary
        self.render_density = render_density    

        self.radius = radius
        
        self.framerate = framerate
        self.SHOW = SHOW

        # All the files from results folder
        if max_frame == -1:
            files_counter = glob.glob(result_dir+'frame_[0-9]*.npy')
            self.max_frame = len(files_counter)
        else:
            self.max_frame = max_frame

        print(self.max_frame)
        # Ordered files:
        self.files = [f"{result_dir}frame_{i}.npy" for i in range(self.max_frame)]


        if render_density:
            self.mass = mass
            self.density_files = [f"{result_dir}frame_density_{i}.npy" for i in range(self.max_frame)]

        if render_boundary:
            b_particles_data = np.load(result_dir+'boundary.npy')
            self.b_particles = ti.Vector.field(3, ti.f32, len(b_particles_data))
            self.b_particles.from_numpy(b_particles_data)
        
        if max_frame == -1:
            self.max_frame = len(self.files)
        else:
            self.max_frame = max_frame

        self.start_frame = start_frame

        self.video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)


    def render(self):
        window = ti.ui.Window("Taichi Fluid Particle Simulation", (1024, 1024),
                            vsync=True)

        canvas = window.get_canvas()
        canvas.set_background_color((1, 1, 1))
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        # draw bounds
        bounds = ti.Vector.field(3, ti.f32, 24)
        bounds[0] = [0, 0, 0]
        bounds[1] = [self.x_bound, 0, 0]
        bounds[2] = [self.x_bound, 0, 0]
        bounds[3] = [self.x_bound, self.y_bound, 0]
        bounds[4] = [self.x_bound, self.y_bound, 0]
        bounds[5] = [0, self.y_bound, 0]
        bounds[6] = [0, self.y_bound, 0]
        bounds[7] = [0, 0, 0]

        bounds[8] = [0, 0, self.z_bound]
        bounds[9] = [self.x_bound, 0, self.z_bound]
        bounds[10] = [self.x_bound, 0, self.z_bound]
        bounds[11] = [self.x_bound, self.y_bound, self.z_bound]
        bounds[12] = [self.x_bound, self.y_bound, self.z_bound]
        bounds[13] = [0, self.y_bound, self.z_bound]
        bounds[14] = [0, self.y_bound, self.z_bound]
        bounds[15] = [0, 0, self.z_bound]

        bounds[16] = [0, 0, 0]
        bounds[17] = [0, 0, self.z_bound]
        bounds[18] = [0, self.y_bound, 0]
        bounds[19] = [0, self.y_bound, self.z_bound]
        bounds[20] = [self.x_bound, 0, 0]
        bounds[21] = [self.x_bound, 0, self.z_bound]
        bounds[22] = [self.x_bound, self.y_bound, 0]
        bounds[23] = [self.x_bound, self.y_bound, self.z_bound]

        for l in range(self.start_frame, self.max_frame):
            camera.position(*self.CAMERA_POS)
            camera.lookat(*CAMERA_LOOK_AT)
            scene.set_camera(camera)

            scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
            scene.ambient_light((0.5, 0.5, 0.5))
            particles_data = np.load(self.files[l])
            particles = ti.Vector.field(3, ti.f32, len(particles_data))
            particles.from_numpy(particles_data)

            if self.render_density:
                density_data = np.load(self.density_files[l])
                density = ti.field(ti.f32, len(density_data))
                density.from_numpy(density_data)

                colors = ti.Vector.field(3, ti.f32, len(particles_data))
                compute_colors_from_density(density, colors)
                scene.particles(centers=particles, radius=self.radius, per_vertex_color=colors)
            else:
                scene.particles(centers=particles, radius=self.radius, color=(0.2, 0.5, 0.8))

            if self.render_boundary:
                scene.particles(centers=self.b_particles, radius=self.radius, color=(0.5, 0.5, 0.5))

            scene.lines(bounds, 1)
            canvas.scene(scene)
            print("Frame: ", l, end="\r")
            self.video_manager.write_frame(window.get_image_buffer_as_numpy())
            if self.SHOW:
                window.show()
