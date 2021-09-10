import taichi as ti
from vector import *
from . import empty_hit_record, set_face_normal
from material import Material
import ray


# struct for sphere
sphere = ti.types.struct(center=Point, radius=ti.f32, material=Material)


def Sphere(center, radius, material):
    return sphere(center=center, radius=radius, material=material)


@ti.func
def hit(sphere, r, t_min, t_max):
    ''' Intersect a ray with a given center and radius.
        Note we pass in the hit record by reference. '''
    hit = False
    rec = empty_hit_record()

    oc = r.orig - sphere.center
    a = r.dir.norm_sqr()
    half_b = oc.dot(r.dir)
    c = oc.norm_sqr() - sphere.radius ** 2

    discriminant = half_b ** 2 - a * c
    # check hit only if discriminint is > 0
    if discriminant >= 0.0:
        sqrtd = ti.sqrt(discriminant)
        root = (-half_b - sqrtd) / a
        hit = (root >= t_min and root < t_max)
        if not hit:
            root = (-half_b + sqrtd) / a
            hit = (root >= t_min and root < t_max)

        if hit:
            rec.t = root
            rec.p = ray.at(r, rec.t)
            outward_normal = (rec.p - sphere.center) / sphere.radius
            set_face_normal(r, outward_normal, rec)

    return hit, rec
