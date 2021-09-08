import taichi as ti


# Wrap taichi types for Vector, etc.
# This provides all the functions we need!
Vector = ti.types.vector(3, ti.f32)
Color = Vector
Point = Vector
