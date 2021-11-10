import taichi as ti
from texture.texture import SOLID
from vector import *
from texture import *
from ray import *


def Checker(even_color, odd_color):
    return Texture(color0=even_color, color1=odd_color, texture_type=CHECKER)


@ti.func
def value(tex_info, u, v, p):
    color = Color(0.0)
    sines = ti.sin(10.0 * p.x) * ti.sin(10.0 * p.y) * ti.sin(10.0 * p.z)
    color = tex_info.color1 if sines < 0 else tex_info.color0
    return color
