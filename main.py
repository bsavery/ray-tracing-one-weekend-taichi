from taichi.lang.ops import random
import taichi as ti
from vector import *
from ray import Ray
from camera import Camera
import time
import random

# First we init taichi.  You can select CPU or GPU, or specify CUDA, Metal, etc
ti.init(arch=ti.gpu)

# imported after ti.init because data structures are created
from texture import *
from texture import commit as texture_commit
from material import *
from hittable import HittableList, Sphere, MovingSphere, HitRecord, XYRect, YZRect, XZRect, Box


def random_scene():
    world = HittableList()

    material_ground = Lambert(Checker(Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9)))
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
                        SolidColor(Color(random.random(), random.random(),
                                         random.random())**2))
                    center2 = center + Vector(0.0, random.random() * 0.5, 0.0)
                    world.add(MovingSphere(center, center2, 0.2, mat, 0.0, 1.0))
                elif choose_mat < 0.95:
                    # metal
                    mat = Metal(
                        SolidColor(Color(random.random(), random.random(),
                                         random.random()) * 0.5 + 0.5),
                        random.random() * 0.5)
                    world.add(Sphere(center, 0.2, mat))
                else:
                    mat = Dielectric(1.5)
                    world.add(Sphere(center, 0.2, mat))

    material_1 = Dielectric(1.5)
    world.add(Sphere(Point(0.0, 1.0, 0.0), 1.0, material_1))

    material_2 = Lambert(SolidColor(Color(0.4, 0.2, 0.1)))
    world.add(Sphere(Point(-4.0, 1.0, 0.0), 1.0, material_2))

    material_3 = Metal(SolidColor(Color(0.7, 0.6, 0.5)), 0.0)
    world.add(Sphere(Point(4.0, 1.0, 0.0), 1.0, material_3))

    return world


def two_spheres():
    world = HittableList()

    mat = Lambert(Checker(Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9)))

    world.add(Sphere(Point(0.0, -10.0, 0.0), 10.0, mat))
    world.add(Sphere(Point(0.0, 10.0, 0.0), 10.0, mat))

    return world


def two_perlin_spheres():
    world = HittableList()

    mat = Lambert(Noise(4))

    world.add(Sphere(Point(0.0, -1000.0, 0.0), 1000.0, mat))
    world.add(Sphere(Point(0.0, 2, 0.0), 2, mat))

    return world


def simple_light():
    world = HittableList()

    mat = Lambert(Noise(4))

    world.add(Sphere(Point(0.0, -1000.0, 0.0), 1000.0, mat))
    world.add(Sphere(Point(0.0, 2, 0.0), 2, mat))

    light_mat = DiffuseLight(SolidColor(Color(4.0, 4.0, 4.0)))
    world.add(XYRect(3, 5, 1, 3, -2, light_mat))

    return world


def earth_sphere():
    world = HittableList()

    mat = Lambert(Image('earthmap.jpg'))
    world.add(Sphere(Point(0.0, 0.0, 0.0), 2, mat))

    return world


def cornell_box():
    world = HittableList()

    red = Lambert(SolidColor(Color(.65, .05, .05)))
    white = Lambert(SolidColor(Color(.73, .73, .73)))
    green = Lambert(SolidColor(Color(.12, .45, .15)))
    light = DiffuseLight(SolidColor(Color(15.0, 15.0, 15.0)))

    world.add(YZRect(0, 555, 0, 555, 555, green))
    world.add(YZRect(0, 555, 0, 555, 0, red))
    world.add(XZRect(213, 343, 227, 332, 554, light))
    world.add(XZRect(0, 555, 0, 555, 0, white))
    world.add(XZRect(0, 555, 0, 555, 555, white))
    world.add(XYRect(0, 555, 0, 555, 555, white))

    world.add(Box(Point(0, 0, 0), Point(165, 330, 165), Point(265.0, 0.0, 295.0), 15.0, white))
    world.add(Box(Point(0, 0, 0), Point(165, 165, 165), Point(130.0, 0.0, 65.0), -18.0, white))

    return world


# Setup image data
ASPECT_RATIO = 16.0 / 9.0
IMAGE_WIDTH = 400
IMAGE_HEIGHT = int(IMAGE_WIDTH / ASPECT_RATIO)
SAMPLES_PER_PIXEL = 100
MAX_DEPTH = 50

INFINITY = 99999999.9


fov = 40.0
aperture = 0.0
scene = 'CORNELL_BOX'
background = Color(0.0)

if scene == 'RANDOM':
    world = random_scene()
    background = Color(0.70, 0.80, 1.00)
    vfrom = Point(13.0, 2.0, 3.0)
    at = Point(0.0, 0.0, 0.0)
    fov = 20.0
    aperture = 0.1
elif scene == 'CHECKER':
    world = two_spheres()
    background = Color(0.70, 0.80, 1.00)
    vfrom = Point(13.0, 2.0, 3.0)
    at = Point(0.0, 0.0, 0.0)
    fov = 20.0
elif scene == 'NOISE':
    world = two_perlin_spheres()
    background = Color(0.70, 0.80, 1.00)
    vfrom = Point(13.0, 2.0, 3.0)
    at = Point(0.0, 0.0, 0.0)
    fov = 20.0
elif scene == 'EARTH':
    world = earth_sphere()
    background = Color(0.70, 0.80, 1.00)
    vfrom = Point(13.0, 2.0, 3.0)
    at = Point(0.0, 0.0, 0.0)
    fov = 20.0
elif scene == 'SIMPLE_LIGHT':
    world = simple_light()
    SAMPLES_PER_PIXEL = 400
    background = Color(0, 0, 0)
    vfrom = Point(26.0, 3.0, 6.0)
    at = Point(0.0, 2.0, 0.0)
    fov = 20.0

elif scene == 'CORNELL_BOX':
    ASPECT_RATIO = 1.0
    IMAGE_WIDTH = 600
    IMAGE_HEIGHT = 600
    world = cornell_box()
    SAMPLES_PER_PIXEL = 400
    background = Color(0, 0, 0)
    vfrom = Point(278.0, 278.0, -800)
    at = Point(278.0, 278.0, 0.0)
    fov = 40.0

up = Vector(0.0, 1.0, 0.0)
focus_dist = 10.0
cam = Camera(vfrom, at, up, fov, ASPECT_RATIO, aperture, focus_dist, 0.0, 1.0)

# This is our pixel array which needs to be setup for the kernel.
# We specify the type and size of the field with 3 channels for RGB
# I set this up with floating point because it will be nicer in the future.
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(IMAGE_WIDTH, IMAGE_HEIGHT))


# A Taichi function that returns a color gradient of the background based on
# the ray direction.
@ti.func
def ray_color(r, world, background):
    color = Color(1.0)  # Taichi functions can only have one return call
    bounces = 1

    # Recursion does not work in taichi so we have to do a while loop
    while bounces < MAX_DEPTH:
        hit, rec, mat_info = world.hit(r, 0.0001, INFINITY)
        if hit:
            emit_color = emit(mat_info, rec)
            scattered, out_ray, attenuation = scatter(mat_info, r, rec)
            if scattered:
                color = emit_color + color * attenuation
                r = out_ray
                bounces += 1
            else:
                color *= emit_color
                break
        else:
            color *= background
            break
    if bounces == MAX_DEPTH:
        color = Color(0.0)

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
        pixels[i, j] += ray_color(ray, world, background)


if __name__ == '__main__':
    world.commit()
    texture_commit()

    gui = ti.GUI("Ray Tracing in One Weekend", res=(IMAGE_WIDTH, IMAGE_HEIGHT))

    first_run = True
    # Run the kernel once for each sample
    for i in range(SAMPLES_PER_PIXEL):
        render_pass()

        gui.set_image(get_buffer(i + 1))
        gui.show()  # show in GUI
        print("\rPercent Complete\t:{:.2%}".format((i + 1)/SAMPLES_PER_PIXEL), end='')
        if first_run:
            t = time.time()
            first_run = False

    print("\nRender time", time.time() - t)
    gui.set_image(get_buffer(SAMPLES_PER_PIXEL))
    gui.show('out.png')
