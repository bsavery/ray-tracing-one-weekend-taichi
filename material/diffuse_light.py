import taichi as ti
from vector import *
from material import *
from ray import *
from texture import *


def DiffuseLight(albedo):
    return Material(albedo=albedo, roughness=0.0, ior=1.0, mat_type=DIFFUSE_LIGHT)


@ti.func
def emit(mat_info, rec):
    '''' Lambrertian scattering '''
    out_color = value(mat_info.albedo, rec.u, rec.v, rec.p)

    return out_color
