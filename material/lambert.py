import taichi as ti
from vector import *
from material import *
from ray import *


def Lambert(color):
    return Material(color=color, roughness=0.0, ior=1.0, mat_type=LAMBERT)


@ti.func
def scatter(mat_info, ray_in, rec):
    '''' Lambrertian scattering '''
    out_direction = rec.normal + random_in_unit_sphere()

    if near_zero(out_direction):
        vec = rec.normal

    return True, Ray(orig=rec.p, dir=out_direction, time=ray_in.time), mat_info.color
