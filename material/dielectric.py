import taichi as ti
from vector import *
from material import *
from ray import *
from texture import *


def Dielectric(ior):
    return Material(albedo=SolidColor(Color(1.0)), roughness=0.0, ior=ior, mat_type=DIELECTRIC)


@ti.func
def scatter(mat_info, ray_in, rec):
    refraction_ratio = 1.0 / mat_info.ior if rec.front_face else mat_info.ior
    unit_dir = ray_in.dir.normalized()
    cos_theta = min(-unit_dir.dot(rec.normal), 1.0)
    sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)

    out_direction = Vector(0.0, 0.0, 0.0)
    cannot_refract = refraction_ratio * sin_theta > 1.0
    if cannot_refract or reflectance(cos_theta, refraction_ratio) > ti.random():
        out_direction = reflect(unit_dir, rec.normal)
    else:
        out_direction = refract(unit_dir, rec.normal, refraction_ratio)
    attenuation = value(mat_info.albedo, rec.u, rec.v, rec.p)

    return True, Ray(orig=rec.p, dir=out_direction, time=ray_in.time), attenuation
