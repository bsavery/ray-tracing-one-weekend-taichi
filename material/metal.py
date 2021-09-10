import taichi as ti
from vector import *
from material import *
from ray import *


def Metal(color, roughness):
    return Material(color=color, roughness=roughness, ior=1.0, mat_type=METAL)


@ti.func
def scatter(mat_info, in_direction, rec):
    ''' Metal scattering '''
    out_direction = reflect(in_direction.normalized(),
                            rec.normal) + mat_info.roughness * random_in_unit_sphere()
    attenuation = mat_info.color
    reflected = out_direction.dot(rec.normal) > 0.0
    return reflected, Ray(orig=rec.p, dir=out_direction), attenuation
