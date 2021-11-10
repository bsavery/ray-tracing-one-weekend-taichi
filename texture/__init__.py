import taichi as ti
from vector import *
from .texture import *
from .solid_color import SolidColor
from .solid_color import value as solid_value
from .checker import Checker
from .checker import value as checker_value

@ti.func
def value(tex_info, u, v, p):
    attenuation = Color(0.0)

    # This may seem like a hack, it's basically to get around taichi not allowing
    # us to pass pointer to objects, like Materials
    if tex_info.texture_type == SOLID:
        attenuation = solid_value(tex_info, u, v, p)
    elif tex_info.texture_type == CHECKER:
        attenuation = checker_value(tex_info, u, v, p)

    return attenuation
