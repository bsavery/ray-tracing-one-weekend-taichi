import taichi as ti
from vector import *
from ray import Ray
import math

@ti.data_oriented
class Camera:
    ''' Camera class '''
    def __init__(self, look_from, look_at, up, vfov, aspect_ratio, aperture, focus_dist, t0, t1):
        theta = math.radians(vfov)
        h = math.tan(theta/2.0)

        # camera position and orientation
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        w = (look_from - look_at).normalized()
        self.u = up.cross(w).normalized()
        self.v = w.cross(self.u)

        self.origin = look_from
        self.horizontal = focus_dist * viewport_width * self.u
        self.vertical = focus_dist * viewport_height * self.v
        self.lower_left_corner = self.origin - self.horizontal/2.0 - \
            self.vertical/2.0 - focus_dist * w

        self.lens_radius = aperture / 2.0
        self.t0, self.t1 = t0, t1


    @ti.func
    def get_ray(self, s, t):
        ''' Computes random sample based on st'''
        rd = self.lens_radius * random_in_unit_disk()
        offset = self.u * rd.x + self.v * rd.y
        return Ray(orig=(self.origin + offset), 
                   dir=(self.lower_left_corner + s*self.horizontal + t*self.vertical - self.origin - offset),
                   time=ti.random() * (self.t1 - self.t0) + self.t0)
