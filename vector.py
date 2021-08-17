import taichi_glsl as ts
import taichi as ti
import math

Vector = ts.vec3
Color = ts.vec3
Point = ts.vec3

WHITE = Color(1.0, 1.0, 1.0)
BLUE = Color(0.5, 0.7, 1.0)
RED = Color(1.0, 0.0, 0.0)


@ti.func
def random_in_unit_disk():
    theta = ti.random() * math.pi * 2.0
    r = ti.random()**0.5

    return Vector(r * ti.cos(theta), r * ti.sin(theta), 0.0)


@ti.func
def random_in_hemisphere(normal):
    vec = random_in_unit_sphere()
    if vec.dot(normal) < 0:
        vec = -vec
    return vec


@ti.func
def random_in_unit_sphere():
    theta = ti.random() * math.pi * 2.0
    v = ti.random()
    phi = ti.acos(2.0 * v - 1.0)
    r = ti.random()**(1 / 3)
    return Vector(r * ti.sin(phi) * ti.cos(theta),
                  r * ti.sin(phi) * ti.sin(theta), r * ti.cos(phi))
