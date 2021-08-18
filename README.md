# ray-tracing-one-weekend-taichi
A fast python implementation of [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) using python and Taichi.

Taichi is a simple "Domain specific language" to use mostly pure python and run on the GPU.  There are other solutions for doing similar things, but my interest in Taichi was it's wide support for platforms, and this was an expirement to see and learn it. Learn more at [Taichi](https://github.com/taichi-dev/taichi).  

![Lots o' balls](https://github.com/bsavery/ray-tracing-one-weekend-taichi/blob/main/out.png?raw=true)

Goes up to Ray tracing "The next week" with the BVH implementation.  I tried to keep the implementation fairly generic to the original text so others can follow along.  There are some details that have to be "vectorized" for Taichi.  For example, rather than a list of Sphere objects, we have a World class which has a list of sphere centers and radii.

To run you will have to do `pip install taichi`

Some notes / thoughts / TODOs.
* I have not implemented more complex memory layouts from taichi such as sparse layouts. 
* More "microkernel" architectures (rather than one big render loop) seem to be slower. But maybe could be optimized.  I tried to have the main kernel do 1 piece of work each pass.  
* Only tested really on macOS.  But performance looks good on metal gpu!  Vulkan and CUDA would be interesting to compare.  For reference this runs in ~9.25 sec on a macbook pro with an AMD 5500M GPU.
