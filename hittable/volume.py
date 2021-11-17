import taichi as ti
from taichi.testing import test
from vector import *
from . import empty_hit_record, set_face_normal
import ray
from material import Material
import math
import sys
from .box import box, hit as box_hit

# struct for sphere
volume = ti.types.struct(boundary=box,
                         neg_inv_density=ti.f32, phase_function=Material,
                         bbox_min=Point, bbox_max=Point)


def Volume(boundary, density, phase_function):
    return volume(boundary=boundary, neg_inv_density=-1.0/density,
                  phase_function=phase_function,
                  bbox_min=boundary.bbox_min, bbox_max=boundary.bbox_max)


@ti.func
def hit(volume, r, t_min, t_max):
    rec = empty_hit_record()
    hit = False

    hit1, rec1, mat1 = box_hit(volume.boundary, r, -999999999.9, 99999999.9)
    if hit1:
        hit2, rec2, mat2 = box_hit(volume.boundary, r, rec1.t + 0.0001, 99999999.9)
        if hit2:
            if (rec1.t < t_min):
                rec1.t = t_min
            if (rec2.t > t_max):
                rec2.t = t_max

            if rec1.t < rec2.t:
                if rec1.t < 0:
                    rec1.t = 0

                ray_length = r.dir.norm()
                distance_inside_boundary = (rec2.t - rec1.t) * ray_length
                hit_distance = volume.neg_inv_density * ti.log(ti.random())

                if hit_distance <= distance_inside_boundary:
                    rec.t = rec1.t + hit_distance / ray_length
                    rec.p = ray.at(r, rec.t)

                    rec.normal = Vector(1, 0, 0)
                    rec.front_face = True
                    hit = True

    return hit, rec, volume.phase_function
