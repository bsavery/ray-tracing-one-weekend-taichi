import taichi as ti
from vector import *
import math


@ti.data_oriented
class Camera:
    def __init__(self, vfrom, at, up, fov, aspect_ratio, aperture, focus_dist):
        theta = math.radians(fov)
        h = math.tan(theta / 2.0)
        viewport_height = 2.0 * h
        viewport_width = viewport_height * aspect_ratio
        focal_length = 1.0

        w = (vfrom - at).normalized()
        u = up.cross(w).normalized()
        v = w.cross(u)

        self.origin = vfrom
        self.horizontal = focus_dist * viewport_width * u
        self.vertical = focus_dist * viewport_height * v
        self.lower_left_corner = self.origin - (self.horizontal / 2.0) \
                                    - (self.vertical / 2.0) \
                                    - focus_dist * w
        self.lens_radius = aperture / 2.0

    @ti.func
    def get_ray(self, u, v):
        rd = self.lens_radius * random_in_unit_disk()
        offset = u * rd.x + v * rd.y
        return self.origin + offset, self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin - offset
