import taichi as ti
import numpy as np
import glob

CAMERA_POS = [1, 1, 1]
CAMERA_LOOK_AT = [0., 0., 0.]

class Renderer:
    def __init__(self, bounds: ti.f32, result_dir: str, radius: ti.f32, max_frame = -1):
        self.bounds = bounds
        self.x_bound = bounds
        self.y_bound = bounds
        self.z_bound = bounds

        self.radius = radius

        # All the files from results folder
        self.files = glob.glob(result_dir+'*.npy')
        
        if max_frame == -1:
            self.max_frame = len(self.files)
        else:
            self.max_frame = max_frame


    def show(self):
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

        for l in range(self.max_frame):
            camera.position(*CAMERA_POS)
            camera.lookat(*CAMERA_LOOK_AT)
            scene.set_camera(camera)

            scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
            scene.ambient_light((0.5, 0.5, 0.5))
            particles_data = np.load(self.files[l])
            print(particles_data)
            particles = ti.Vector.field(3, ti.f32, len(particles_data))
            particles.from_numpy(particles_data)
            scene.particles(centers=particles, radius=self.radius)
            scene.lines(bounds, 1)
            canvas.scene(scene)
            print("Frame: ", l, end="\r")
            window.show()


    # Has to be implemented
    def render_frame(self, frame: int):
        pass


    def render(self):
        pass