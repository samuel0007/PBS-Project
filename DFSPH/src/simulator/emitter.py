import numpy as np
import taichi as ti
import random

class Emitter:
    def __init__(self, pos, radius, seed = 42):
        """The particle Emitter generates particle positions uniformly sampled over a circle that is parallel to the xz-plane"""
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.radius = radius
        self. err = 0.
        random.seed(seed)

    def get_particle(self):
        angle = random.uniform(0, 2 * np.pi)
        r = np.sqrt(random.uniform(0, self.radius**2))
        part_x = self.x + np.cos(angle) * r
        part_y = self.y 
        part_z = self.z + np.sin(angle) * r
        arr = ti.Vector([part_x, part_y, part_z])
        # np.array([part_x, part_y, part_z], dtype = float)
        return arr
    
    def emit_particles(self, dt, particles_per_second):
        """Returns a list of particle positions. Returns particles_per_second * dt particles, while adding the error to the next call."""
        num_frac = (particles_per_second * dt) + self.err
        num = int(np.floor(num_frac))
        self.err = num_frac - num
        particles = []
        for i in range(num):
            particles.append(self.get_particle())
        return particles

def main():
    emitter = Emitter(0,0,0,1.)
    num_emitted = 0
    for i in range(200):
        num_emitted += len(emitter.emit_particles(0.005,192))
    print(num_emitted)

if __name__ == '__main__':
    main()
