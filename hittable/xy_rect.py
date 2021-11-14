import taichi as ti
from vector import *
from . import empty_hit_record, set_face_normal
from material import Material
import ray


# struct for sphere
xy_rect = ti.types.struct(x0=ti.f32, y0=ti.f32, x1=ti.f32, y1=ti.f32, k=ti.f32, material=Material)


def XYRect(x0, x1, y0, y1, k, material):
    return xy_rect(x0=x0, y0=y0, x1=x1, y1=y1, k=k, material=material)


@ti.func
def hit(xy_rect, r, t_min, t_max):
    ''' Intersect a ray with a given center and radius.
        Note we pass in the hit record by reference. '''
    hit = False
    rec = empty_hit_record()

    t = (xy_rect.k-r.orig.z) / r.dir.z
    if t >= t_min and t <= t_max:
        x = r.orig.x + t*r.dir.x
        y = r.orig.y + t*r.dir.y

        if x >= xy_rect.x0 and x <= xy_rect.x1 and y >= xy_rect.y0 and y <= xy_rect.y1:
            rec.u = (x-xy_rect.x0)/(xy_rect.x1-xy_rect.x0)
            rec.v = (y-xy_rect.y0)/(xy_rect.y1-xy_rect.y0)
            rec.t = t
            outward_normal = Vector([0, 0, 1])
            set_face_normal(r, outward_normal, rec)
            rec.p = ray.at(r, rec.t)
            hit = True

    return hit, rec
