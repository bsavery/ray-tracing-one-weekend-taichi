import taichi as ti
from vector import *


# struct containing all the material info needed to run scatter functions
Material = ti.types.struct(color=Color, roughness=ti.f32, ior=ti.f32, mat_type=ti.i32)


@ti.func
def empty_material():
    ''' Constructs an empty material info set'''
    return Material(color=Color(0.0), roughness=0.0, ior=0.0, mat_type=0)


# material type constants
LAMBERT = 0
METAL = 1
DIELECTRIC = 2
