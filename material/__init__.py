import taichi as ti
from vector import *
from .funcs import *
from .material import *
from .lambert import Lambert
from .lambert import scatter as lambert_scatter
from .metal import Metal
from .metal import scatter as metal_scatter
from .dielectric import Dielectric
from .dielectric import scatter as dielectric_scatter
from ray import *


@ti.func
def scatter(mat_info, ray_in, rec):
    scatter = False
    out_ray = Ray(orig=Point(0.0), dir=Vector(0.0), time=0.0)
    attenuation = Color(0.0)

    # This may seem like a hack, it's basically to get around taichi not allowing
    # us to pass pointer to objects, like Materials
    if mat_info.mat_type == LAMBERT:
        scatter, out_ray, attenuation = lambert_scatter(mat_info, ray_in, rec)
    elif mat_info.mat_type == METAL:
        scatter, out_ray, attenuation = metal_scatter(mat_info, ray_in, rec)
    else:
        scatter, out_ray, attenuation = dielectric_scatter(mat_info, ray_in, rec)

    return scatter, out_ray, attenuation
