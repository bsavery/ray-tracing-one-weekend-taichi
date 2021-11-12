import taichi as ti
from texture.texture import SOLID
from vector import *
from texture import *
from ray import *
import random

POINT_COUNT = 256
NOISE_INITTED = False
RAND_FLOAT = ti.field(ti.f32, shape=1)
PERM = ti.Vector.field(3, ti.i32, shape=1)
RAND_VEC = ti.Vector.field(3, dtype=ti.f32, shape=1)


def init_noise():
    global NOISE_INITTED
    if NOISE_INITTED:
        return
    else:
        NOISE_INITTED = True

    global RAND_FLOAT
    global PERM
    global RAND_VEC
    RAND_FLOAT = ti.field(ti.f32, shape=POINT_COUNT)
    PERM = ti.Vector.field(3, dtype=ti.i32, shape=POINT_COUNT)
    RAND_VEC = ti.Vector.field(3, dtype=ti.f32, shape=POINT_COUNT)

    # initial values
    for i in range(POINT_COUNT):
        RAND_FLOAT[i] = random.random()
        PERM[i] = [i, i, i]
        RAND_VEC[i] = [random.random() * 2 - 1.0, random.random() * 2 - 1.0, random.random() * 2 - 1.0]

    i = 255
    while i >= 0:
        for j in range(3):
            target = random.randint(0, i)
            tmp = PERM[i][j]
            PERM[i][j] = PERM[target][j]
            PERM[target][j] = tmp
        i -= 1


def Noise(scale):
    init_noise()
    return Texture(color0=Color(scale), color1=Color(0.0), texture_type=NOISE)


@ti.func
def noise(p):
    u = p.x - ti.floor(p.x)
    v = p.y - ti.floor(p.y)
    w = p.z - ti.floor(p.z)

    i = ti.cast(ti.floor(p.x), ti.i32)
    j = ti.cast(ti.floor(p.y), ti.i32)
    k = ti.cast(ti.floor(p.z), ti.i32)

    c = ti.Matrix([[0.0, 0.0, 0.0]] * 8)
    for di in ti.static(range(2)):
        for dj in ti.static(range(2)):
            for dk in ti.static(range(2)):
                vec = RAND_VEC[PERM[(i+di) & 255].x ^
                               PERM[(j+dj) & 255].y ^
                               PERM[(k+dk) & 255].z]
                c[di*4 + dj*2 + dk, 0] = vec[0]
                c[di*4 + dj*2 + dk, 1] = vec[1]
                c[di*4 + dj*2 + dk, 2] = vec[2]

    return perlin_interp(c, u, v, w)


@ti.func
def perlin_interp(c, u, v, w):
    accum = 0.0
    uu = u*u*(3-2*u)
    vv = v*v*(3-2*v)
    ww = w*w*(3-2*w)

    for i in ti.static(range(2)):
        for j in ti.static(range(2)):
            for k in ti.static(range(2)):
                weight_v = ti.Vector([u-i, v-j, w-k])
                c_vec = ti.Vector([c[i*4 + j*2 + k, 0], c[i*4 + j*2 + k, 1], c[i*4 + j*2 + k, 2]])
                accum += (i*uu + (1-i)*(1-uu)) \
                    * (j*vv + (1-j)*(1-vv)) \
                    * (k*ww + (1-k)*(1-ww)) \
                    * weight_v.dot(c_vec)

    return accum


@ti.func
def turb(p):
    accum = 0.0
    temp_p = p
    weight = 1.0

    for i in range(7):
        accum += weight*noise(temp_p)
        weight *= 0.5
        temp_p *= 2

    return ti.abs(accum)


@ti.func
def value(tex_info, u, v, p):
    turb_val = 10.0 * turb(p)
    scale = tex_info.color0[0]
    return Color(0.5 * (1.0 + ti.sin(scale * p.z + turb_val)))
