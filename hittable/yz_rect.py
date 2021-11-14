import taichi as ti
from vector import *
from . import empty_hit_record, set_face_normal
from material import Material
import ray


# struct for sphere
yz_rect = ti.types.struct(y0=ti.f32, z0=ti.f32, y1=ti.f32, z1=ti.f32, k=ti.f32, material=Material)


def YZRect(y0, y1, z0, z1, k, material):
    return yz_rect(y0=y0, z0=z0, y1=y1, z1=z1, k=k, material=material)


@ti.func
def hit(yz_rect, r, t_min, t_max):
    ''' Intersect a ray with a given center and radius.
        Note we pass in the hit record by reference. '''
    hit = False
    rec = empty_hit_record()

    t = (yz_rect.k-r.orig.x) / r.dir.x
    if t >= t_min and t <= t_max:
        y = r.orig.y + t*r.dir.y
        z = r.orig.z + t*r.dir.z

        if y >= yz_rect.y0 and y <= yz_rect.y1 and z >= yz_rect.z0 and z <= yz_rect.z1:
            rec.u = (y-yz_rect.y0)/(yz_rect.y1-yz_rect.y0)
            rec.v = (y-yz_rect.z0)/(yz_rect.z1-yz_rect.z0)
            rec.t = t
            outward_normal = Vector([1, 0, 0])
            set_face_normal(r, outward_normal, rec)
            rec.p = ray.at(r, rec.t)
            hit = True

    return hit, rec
