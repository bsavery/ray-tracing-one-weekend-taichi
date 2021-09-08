import taichi as ti
from vector import *


# a ray class
Ray = ti.types.struct(orig=Point, dir=Vector)


@ti.func
def at(r, t):
    ''' Computes the point of ray at t '''
    return r.orig + r.dir * t
