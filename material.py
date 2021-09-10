import taichi as ti
from taichi_glsl.vector import reflect
from vector import *
from ray import Ray

# struct containing all the material info needed to run scatter functions
Material = ti.types.struct(color=Color, roughness=ti.f32, ior=ti.f32, mat_type=ti.i32)

# material type constants
LAMBERT = 0
METAL = 1
DIELECTRIC = 2

@ti.func
def empty_material():
    ''' Constructs an empty material info set'''
    return Material(color=Color(0.0), roughness=0.0, ior=0.0, mat_type=0)


@ti.func
def reflectance(cosine, idx):
    r0 = ((1.0 - idx) / (1.0 + idx))**2
    return r0 + (1.0 - r0) * ((1.0 - cosine)**5)


@ti.func
def reflect(v, n):
    return v - 2.0 * v.dot(n) * n


@ti.func
def refract(v, n, etai_over_etat):
    cos_theta = min(-v.dot(n), 1.0)
    r_out_perp = etai_over_etat * (v + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.norm_sqr())) * n
    return r_out_perp + r_out_parallel


def Lambert(color):
    return Material(color=color, roughness=0.0, ior=1.0, mat_type=LAMBERT)

@ti.func
def lambert_scatter(mat_info, in_direction, rec):
    out_direction = rec.normal + random_in_unit_sphere()

    if near_zero(out_direction):
        vec = rec.normal

    return True, Ray(orig=rec.p, dir=out_direction), mat_info.color


def Metal(color, roughness):
    return Material(color=color, roughness=roughness, ior=1.0, mat_type=METAL)


@ti.func
def metal_scatter(mat_info, in_direction, rec):
    out_direction = reflect(in_direction.normalized(),
                            rec.normal) + mat_info.roughness * random_in_unit_sphere()
    attenuation = mat_info.color
    reflected = out_direction.dot(rec.normal) > 0.0
    return reflected, Ray(orig=rec.p, dir=out_direction), attenuation


def Dielectric(ior):
    return Material(color=Color(1.0), roughness=0.0, ior=ior, mat_type=DIELECTRIC)

@ti.func
def dielectric_scatter(mat_info, in_direction, rec):
    refraction_ratio = 1.0 / mat_info.ior if rec.front_face else mat_info.ior
    unit_dir = in_direction.normalized()
    cos_theta = min(-unit_dir.dot(rec.normal), 1.0)
    sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)

    out_direction = Vector(0.0, 0.0, 0.0)
    cannot_refract = refraction_ratio * sin_theta > 1.0
    if cannot_refract or reflectance(cos_theta, refraction_ratio) > ti.random():
        out_direction = reflect(unit_dir, rec.normal)
    else:
        out_direction = refract(unit_dir, rec.normal, refraction_ratio)
    attenuation = mat_info.color

    return True, Ray(orig=rec.p, dir=out_direction), attenuation


@ti.func
def scatter(mat_info, in_direction, rec):
    scatter = False
    out_ray = Ray(orig=Point(0.0), dir=Vector(0.0))
    attenuation = Color(0.0)

    # This may seem like a hack, it's basically to get around taichi not allowing
    # us to pass pointer to objects, like Materials
    if mat_info.mat_type == LAMBERT:
        scatter, out_ray, attenuation = lambert_scatter(mat_info, in_direction, rec)
    elif mat_info.mat_type == METAL:
        scatter, out_ray, attenuation = metal_scatter(mat_info, in_direction, rec)
    else:
        scatter, out_ray, attenuation = dielectric_scatter(mat_info, in_direction, rec)

    return scatter, out_ray, attenuation
