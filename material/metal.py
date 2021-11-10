import taichi as ti
from vector import *
from material import *
from ray import *
from texture import *


def Metal(albedo, roughness):
    return Material(albedo=albedo, roughness=roughness, ior=1.0, mat_type=METAL)


@ti.func
def scatter(mat_info, ray_in, rec):
    ''' Metal scattering '''
    out_direction = reflect(ray_in.dir.normalized(),
                            rec.normal) + mat_info.roughness * random_in_unit_sphere()
    attenuation = value(mat_info.albedo, rec.u, rec.v, rec.p)
    reflected = out_direction.dot(rec.normal) > 0.0
    return reflected, Ray(orig=rec.p, dir=out_direction, time=ray_in.time), attenuation
