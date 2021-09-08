import taichi as ti


# First we init taichi.  You can select CPU or GPU, or specify CUDA, Metal, etc
ti.init(arch=ti.gpu)

# width and height of the image
IMAGE_WIDTH, IMAGE_HEIGHT = 256, 256

# This is our pixel array which needs to be setup for the kernel.
# We specify the type and size of the field with 3 channels for RGB
# I set this up with floating point because it will be nicer in the future.
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(IMAGE_WIDTH, IMAGE_HEIGHT))

# Our "kernel".  This loops over all the pixels in a parallel manner
# We don't multiply by 256 as in the original code since we use floats
@ti.kernel
def fill_pixels():
    for i, j in pixels:
        pixels[i, j] = [i / (IMAGE_WIDTH - 1), j / (IMAGE_HEIGHT - 1), 0.25]


if __name__ == '__main__':
    gui = ti.GUI("Ray Tracing in One Weekend", res=(IMAGE_WIDTH, IMAGE_HEIGHT))

    # Run the kernel
    fill_pixels()

    gui.set_image(pixels)
    gui.show("out.png")  # export and show in GUI

