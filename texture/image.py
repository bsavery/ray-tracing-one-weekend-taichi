import taichi as ti
from texture.texture import SOLID
from vector import *
from texture import *
from ray import *
import random
import numpy as np
from PIL import Image as pil_image

TEXTURE_CACHE_ARR = []


def read_texture_file(filename):
    ''' Read a texture file from disk and save contents to cache array.
        return width, heigh, start pointer '''
    global TEXTURE_CACHE_ARR
    img = pil_image.open(filename)
    img_data = np.array(img)

    height, width, _ = img_data.shape
    img_data = img_data.reshape((width * height, 3)).astype(np.single)
    img_data /= 255.0

    if TEXTURE_CACHE_ARR == []:
        ptr = 0
        TEXTURE_CACHE_ARR = img_data
    else:
        ptr = len(TEXTURE_CACHE_ARR)
        TEXTURE_CACHE_ARR = np.concatenate((TEXTURE_CACHE_ARR, img_data), axis=0)

    TEXTURE_CACHE_ARR = img_data
    return height, width, ptr


def commit():
    global TEXTURE_CACHE
    global TEXTURE_CACHE_ARR
    if TEXTURE_CACHE_ARR != []:
        TEXTURE_CACHE = ti.Vector.field(3, ti.f32, shape=len(TEXTURE_CACHE_ARR))
        TEXTURE_CACHE.from_numpy(TEXTURE_CACHE_ARR)
    else:
        TEXTURE_CACHE = ti.Vector.field(3, ti.f32, shape=1)


def Image(filename):
    width, height, ptr = read_texture_file(filename)
    return Texture(color0=Color(width, height, ptr), color1=Color(0.0), texture_type=TEXTURE)


@ti.func
def clamp(x, min_x, max_x):
    return max(min(x, max_x), min_x)


@ti.func
def value(tex_info, u, v, p):
    image_info = tex_info.color0
    width = ti.cast(image_info[0], ti.i32)
    height = ti.cast(image_info[1], ti.i32)
    image_ptr = ti.cast(image_info[2], ti.i32)

    u = clamp(u, 0.0, 1.0)
    v = 1.0 - clamp(v, 0.0, 1.0)  # v gets flipped

    i = ti.cast(u * width, ti.i32)
    j = ti.cast(v * height, ti.i32)

    if i >= width:
        i = width - 1
    if j >= height:
        j = height - 1

    pixel = TEXTURE_CACHE[image_ptr + j * width + i]
    return pixel
