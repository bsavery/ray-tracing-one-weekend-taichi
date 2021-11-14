import taichi as ti
from vector import *
from texture import *


# struct containing all the material info needed to run scatter functions
Material = ti.types.struct(albedo=Texture, roughness=ti.f32, ior=ti.f32, mat_type=ti.i32)


@ti.func
def empty_material():
    ''' Constructs an empty material info set'''
    return Material(albedo=empty_texture(), roughness=0.0, ior=0.0, mat_type=0)


# material type constants
LAMBERT = 0
METAL = 1
DIELECTRIC = 2
DIFFUSE_LIGHT = 3
