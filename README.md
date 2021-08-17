# ray-tracing-one-weekend-taichi
A fast python implementation of Ray Tracing in One Weekend using python and Taichi

![Lots o' balls](https://github.com/bsavery/ray-tracing-one-weekend-taichi/blob/[branch]/out.png?raw=true)

Goes up to Ray tracing "The next week" with the BVH implementation.

Some notes / thoughts / TODOs.
I have not implemented more complex data layouts from taichi or sparse layouts. 
More "microkernel" architectures seem to be slower. But maybe could be optimized.
Only tested really on macOS.  But performance looks good!  Vulkan and CUDA would be interesting to compare.  