import numpy as np
import taichi as ti
import random

class Emitter:
    def __init__(self, x, y, z, radius):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius

    def get_particle(self):
        angle = random.uniform(0, 2 * np.pi)
        r = np.sqrt(random.uniform(0, self.radius**2))
        part_x = self.x + np.cos(angle) * r
        part_y = self.y 
        part_z = self.z + np.sin(angle) * r
        arr = ti.Vector([part_x, part_y, part_z])
        # np.array([part_x, part_y, part_z], dtype = float)
        return arr

def main():
    emitter = Emitter(0,0,0,1.)
    for i in range(10):
        print(emitter.get_particle())

if __name__ == '__main__':
    main()
