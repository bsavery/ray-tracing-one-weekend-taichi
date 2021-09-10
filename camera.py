import taichi as ti
from vector import *
from ray import Ray
import math

@ti.data_oriented
class Camera:
    ''' Camera class '''
    def __init__(self, look_from, look_at, up, vfov, aspect_ratio):
        theta = math.radians(vfov)
        h = math.tan(theta/2.0)

        # camera position and orientation
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        w = (look_from - look_at).normalized()
        u = up.cross(w).normalized()
        v = w.cross(u)

        self.origin = look_from
        self.horizontal = viewport_width * u
        self.vertical = viewport_height * v
        self.lower_left_corner = self.origin - self.horizontal/2.0 - \
            self.vertical/2.0 - w


    @ti.func
    def get_ray(self, s, t):
        ''' Computes random sample based on uv'''
        return Ray(orig=self.origin, dir=(self.lower_left_corner + s*self.horizontal +
                                          t*self.vertical - self.origin))
