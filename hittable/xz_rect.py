import taichi as ti
from vector import *
from . import empty_hit_record, set_face_normal
from material import Material
import ray


# struct for sphere
xz_rect = ti.types.struct(x0=ti.f32, z0=ti.f32, x1=ti.f32, z1=ti.f32, k=ti.f32, material=Material,
                          bbox_min=Point, bbox_max=Point)


def XZRect(x0, x1, z0, z1, k, material):
    return xz_rect(x0=x0, z0=z0, x1=x1, z1=z1, k=k, material=material,
                   bbox_min=Point([x0, k-0.0001, z0]),
                   bbox_max=Point([x1, k+0.0001, z1]))


@ti.func
def hit(xz_rect, r, t_min, t_max):
    ''' Intersect a ray with a given center and radius.
        Note we pass in the hit record by reference. '''
    hit = False
    rec = empty_hit_record()

    t = (xz_rect.k-r.orig.y) / r.dir.y
    if t >= t_min and t <= t_max:
        x = r.orig.x + t*r.dir.x
        z = r.orig.z + t*r.dir.z

        if x >= xz_rect.x0 and x <= xz_rect.x1 and z >= xz_rect.z0 and z <= xz_rect.z1:
            rec.u = (x-xz_rect.x0)/(xz_rect.x1-xz_rect.x0)
            rec.v = (z-xz_rect.z0)/(xz_rect.z1-xz_rect.z0)
            rec.t = t
            outward_normal = Vector([0, 1, 0])
            set_face_normal(r, outward_normal, rec)
            rec.p = ray.at(r, rec.t)
            hit = True

    return hit, rec, xz_rect.material
