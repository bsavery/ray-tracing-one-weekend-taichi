from taichi.lang.ops import random
from hittable import HittableList, Sphere, HitRecord
import taichi as ti
from vector import *
from ray import Ray
from camera import Camera
import time
from material import *
import random

# First we init taichi.  You can select CPU or GPU, or specify CUDA, Metal, etc
ti.init(arch=ti.gpu)


def random_scene():
    world = HittableList()

    material_ground = Lambert(Color(0.5, 0.5, 0.5))
    world.add(Sphere(Point(0.0, -1000.0, 0.0), 1000.0, material_ground))

    static_point = Point(4.0, 0.2, 0.0)
    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.random()
            center = Point(a + 0.9 * random.random(), 0.2,
                           b + 0.9 * random.random())

            if (center - static_point).norm() > 0.9:
                if choose_mat < 0.8:
                    # diffuse
                    mat = Lambert(
                        Color(random.random(), random.random(),
                              random.random())**2)
                elif choose_mat < 0.95:
                    # metal
                    mat = Metal(
                        Color(random.random(), random.random(),
                              random.random()) * 0.5 + 0.5,
                        random.random() * 0.5)
                else:
                    mat = Dielectric(1.5)

            world.add(Sphere(center, 0.2, mat))

    material_1 = Dielectric(1.5)
    world.add(Sphere(Point(0.0, 1.0, 0.0), 1.0, material_1))

    material_2 = Lambert(Color(0.4, 0.2, 0.1))
    world.add(Sphere(Point(-4.0, 1.0, 0.0), 1.0, material_2))

    material_3 = Metal(Color(0.7, 0.6, 0.5), 0.0)
    world.add(Sphere(Point(4.0, 1.0, 0.0), 1.0, material_3))

    return world


# Setup image data
ASPECT_RATIO = 3.0 / 2.0
IMAGE_WIDTH = 1200
IMAGE_HEIGHT = int(IMAGE_WIDTH / ASPECT_RATIO)
SAMPLES_PER_PIXEL = 500
MAX_DEPTH = 50

INFINITY = 99999999.9

# This is our pixel array which needs to be setup for the kernel.
# We specify the type and size of the field with 3 channels for RGB
# I set this up with floating point because it will be nicer in the future.
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(IMAGE_WIDTH, IMAGE_HEIGHT))

world = random_scene()

vfrom = Point(13.0, 2.0, 3.0)
at = Point(0.0, 0.0, 0.0)
up = Vector(0.0, 1.0, 0.0)
focus_dist = 10.0
aperture = 0.1
cam = Camera(vfrom, at, up, 20.0, ASPECT_RATIO, aperture, focus_dist)

# A Taichi function that returns a color gradient of the background based on
# the ray direction.
@ti.func
def ray_color(r, world):
    color = Color(1.0)  # Taichi functions can only have one return call
    bounces = 1

    # Recursion does not work in taichi so we have to do a while loop
    while bounces < MAX_DEPTH:
        hit, rec, mat_info = world.hit(r, 0.0001, INFINITY)
        if hit:
            scattered, out_ray, attenuation = scatter(mat_info, r.dir, rec)
            if scattered:
                color *= attenuation
                r = out_ray
                bounces += 1
            else:
                color = Color(0.0)
                break
        else:
            unit_direction = r.dir.normalized()
            t = 0.5 * (unit_direction.y + 1.0)
            color = color * ((1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0))
            break
    return color


def get_buffer(samples):
    ''' Do gamma and divide accumulated buffer by samples '''
    return (pixels.to_numpy() / samples) ** 0.5


# Our "kernel".  This loops over all the samples for a pixel in a parallel manner
# We don't multiply by 256 as in the original code since we use floats
@ti.kernel
def render_pass():
    for i, j in pixels:
        u, v = (i + ti.random()) / (IMAGE_WIDTH - 1), (j + ti.random()) / (IMAGE_HEIGHT - 1)
        ray = cam.get_ray(u, v)
        pixels[i, j] += ray_color(ray, world)


if __name__ == '__main__':
    world.commit()

    gui = ti.GUI("Ray Tracing in One Weekend", res=(IMAGE_WIDTH, IMAGE_HEIGHT))

    t = time.time()
    # Run the kernel once for each sample
    for i in range(SAMPLES_PER_PIXEL):
        render_pass()

        gui.set_image(get_buffer(i + 1))
        gui.show()  # show in GUI
        print("\rPercent Complete\t:{:.2%}".format((i + 1)/SAMPLES_PER_PIXEL), end='')

    print("\nRender time", time.time() - t)
    gui.set_image(get_buffer(SAMPLES_PER_PIXEL))
    gui.show('out.png')
