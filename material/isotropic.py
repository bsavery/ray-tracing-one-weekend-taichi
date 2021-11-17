import taichi as ti
from vector import *
from material import *
from ray import *
from texture import *


def Isotropic(albedo):
    return Material(albedo=albedo, roughness=0.0, ior=0.0, mat_type=ISOTROPIC)


@ti.func
def scatter(mat_info, ray_in, rec):
    attenuation = value(mat_info.albedo, rec.u, rec.v, rec.p)

    return True, Ray(orig=rec.p, dir=random_in_unit_sphere(), time=ray_in.time), attenuation
