import taichi as ti
from vector import *


# struct containing all the texture info needed to run value function
Texture = ti.types.struct(color0=Color, color1=Color, texture_type=ti.i32)


@ti.func
def empty_texture():
    ''' Constructs an empty texture info set'''
    return Texture(color0=Color(0.0), color1=Color(0.0), texture_type=0)


# texture type constants
SOLID = 0
CHECKER = 1
