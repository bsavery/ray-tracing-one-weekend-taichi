import taichi as ti
import taichi_glsl

# Wrap taichi types for Vector, etc.
# This provides all the functions we need!
Vector = ti.types.vector(3, ti.f32)
Color = Vector
Point = Vector


@ti.func
def random_in_unit_sphere():
    # handily, this function returns a normalized vec
    return taichi_glsl.randgen.randUnit3D()

@ti.func
def random_in_hemi_sphere(normal):
    vec = taichi_glsl.randgen.randUnit3D()
    if vec.dot(normal) <= 0.0:
        vec = -vec

    return vec


@ti.func
def near_zero(vec):
    eps = 1e-8
    return abs(vec.x) < eps and abs(vec.y) < eps and abs(vec.z) < eps
