import taichi as ti
from vector import *
from ray import Ray

@ti.data_oriented
class Camera:
    ''' Camera class '''
    def __init__(self, aspect_ratio):
        # camera position and orientation
        viewport_height = 2.0
        viewport_width = aspect_ratio * viewport_height
        focal_length = 1.0

        self.origin = Point(0.0, 0.0, 0.0)
        self.horizontal = Vector(viewport_width, 0.0, 0.0)
        self.vertical = Vector(0, viewport_height, 0.0)
        self.lower_left_corner = self.origin - self.horizontal/2.0 - \
            self.vertical/2.0 - Vector(0.0, 0.0, focal_length)


    @ti.func
    def get_ray(self, u, v):
        ''' Computes random sample based on uv'''
        return Ray(orig=self.origin, dir=(self.lower_left_corner + u*self.horizontal +
                                          v*self.vertical - self.origin))
