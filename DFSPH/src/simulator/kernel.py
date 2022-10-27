import taichi as ti
import math

@ti.data_oriented
class Poly6:
    def __init__(self, support_radius: ti.f32):
        self.support_radius = support_radius
        self.support_radius2 = support_radius**2
        self.k = 315. / (64. * math.pi * support_radius**9)
        self.l = -945. / (32. * math.pi * support_radius**9)

    @ti.func
    def W(self, r: ti.types.vector(3, ti.f32)) -> ti.f32:
        res = 0.
        r_norm2 = r.norm_sqr()
        if r_norm2 < self.support_radius2:
            res = self.k * (self.support_radius2 - r_norm2)**3
        return res

    @ti.func
    def W_grad(self, r: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
        res = ti.Vector([0., 0., 0.])
        r_norm2 = r.norm_sqr()
        if r_norm2 < self.support_radius2:
            res = self.l * (self.support_radius2 - r_norm2)**2 * r
        return res

@ti.data_oriented
class Spiky:
    def __init__(self, support_radius: ti.f32):
        self.support_radius = support_radius
        self.support_radius2 = support_radius**2
        self.k = 15. / (math.pi * support_radius**6)
        self.l = -45. / (math.pi * support_radius**6)

    @ti.func
    def W(self, r: ti.types.vector(3, ti.f32)) -> ti.f32:
        res = 0.
        r_norm = r.norm()
        if r_norm**2 < self.support_radius2:
            res = self.k * (self.support_radius - r_norm)**3
        return res
    
    @ti.func
    def W_grad(self, r: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
        res = ti.Vector([0., 0., 0.])
        r_norm = r.norm()
        if r_norm > 1e-6 and r_norm**2 < self.support_radius2:
            res = self.l * (self.support_radius - r_norm)**2 * r / r_norm
        return res

@ti.data_oriented
class CubicSpline:
    def __init__(self, support_radius: ti.f32):
        self.support_radius = support_radius
        self.support_radius3 = support_radius**3

        self.k = 8. / (math.pi * self.support_radius3)
        self.l = 48. / (math.pi * self.support_radius3)

    @ti.func
    def W(self, r: ti.types.vector(3, ti.f32)) -> ti.f32:
        res = 0.
        r_norm = r.norm()
        q = r_norm / self.support_radius
        if q <= 1.:
            if q <= 0.5:
                res = self.k * (6. * q**3 - 6. * q**2 + 1.)
            else:
                res = self.k * (2. * (1. - q)**3)
        return res
    
    @ti.func
    def W_grad(self, r: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
        res = ti.Vector([0., 0., 0.])
        r_norm = r.norm()
        q = r_norm / self.support_radius
        if r_norm > 1e-5 and q <= 1.:
            gradq = r * (1. / (r_norm * self.support_radius))
            if q <= 0.5:
                res = self.l*q*(3.*q-2.)*gradq
            else:
                res = self.l*((q-1.)*(1.-q))*gradq
        return res