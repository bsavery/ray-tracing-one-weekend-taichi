import taichi as ti
from vector import *
from ray import Ray


# First we init taichi.  You can select CPU or GPU, or specify CUDA, Metal, etc
ti.init(arch=ti.gpu)

# Setup image data
ASPECT_RATIO = 16.0 / 9.0
IMAGE_WIDTH = 400
IMAGE_HEIGHT = int(IMAGE_WIDTH / ASPECT_RATIO)

# camera position and orientation
VIEWPORT_HEIGHT = 2.0
VIEWPORT_WIDTH = ASPECT_RATIO * VIEWPORT_HEIGHT
FOCAL_LENGTH = 1.0

ORIGIN = Point(0.0, 0.0, 0.0)
HORIZONTAL = Vector(VIEWPORT_WIDTH, 0.0, 0.0)
VERTICAL = Vector(0, VIEWPORT_HEIGHT, 0.0)
LOWER_LEFT_CORNER = ORIGIN - HORIZONTAL/2.0 - VERTICAL/2.0 - Vector(0.0, 0.0, FOCAL_LENGTH)

# This is our pixel array which needs to be setup for the kernel.
# We specify the type and size of the field with 3 channels for RGB
# I set this up with floating point because it will be nicer in the future.
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(IMAGE_WIDTH, IMAGE_HEIGHT))


# A Taichi function that returns a color gradient of the background based on
# the ray direction
@ti.func
def ray_color(r):
    unit_direction = r.dir.normalized()
    t = 0.5 * (unit_direction.y + 1.0)
    return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0)


# Our "kernel".  This loops over all the pixels in a parallel manner
# We don't multiply by 256 as in the original code since we use floats
@ti.kernel
def fill_pixels():
    for i, j in pixels:
        u, v = i / (IMAGE_WIDTH - 1), j / (IMAGE_HEIGHT - 1)
        ray = Ray(orig=ORIGIN, dir=(LOWER_LEFT_CORNER + u * HORIZONTAL + v * VERTICAL - ORIGIN))
        pixels[i, j] = ray_color(ray)


if __name__ == '__main__':
    gui = ti.GUI("Ray Tracing in One Weekend", res=(IMAGE_WIDTH, IMAGE_HEIGHT))

    # Run the kernel
    fill_pixels()

    gui.set_image(pixels)
    gui.show("out.png")  # export and show in GUI

