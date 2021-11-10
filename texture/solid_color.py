import taichi as ti
from texture.texture import SOLID
from vector import *
from texture import *
from ray import *


def SolidColor(color):
    return Texture(color0=color, color1=Color(0.0), texture_type=SOLID)


@ti.func
def value(tex_info, u, v, p):
    return tex_info.color0
