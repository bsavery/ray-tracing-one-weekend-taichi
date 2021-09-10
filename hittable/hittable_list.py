import taichi as ti
from vector import *
from .hit_record import empty_hit_record
from .sphere import sphere
from .sphere import hit as hit_sphere
from material import empty_material


@ti.data_oriented
class HittableList:
    def __init__(self):
        self.objects = []

    def add(self, object):
        self.objects.append(object)

    def commit(self):
        ''' Save the sphere data and material info so we can loop over these.'''
        self.n = len(self.objects)
        self.spheres = sphere.field(shape=(self.n,))

        for i, obj in enumerate(self.objects):
            self.spheres[i] = obj

    @ti.func
    def hit(self, r, t_min, t_max):
        hit_anything = False
        closest_so_far = t_max
        rec = empty_hit_record()
        mat_info = empty_material()

        for i in range(self.n):
            hit, temp_rec = hit_sphere(self.spheres[i], r, t_min, closest_so_far)
            if hit:
                hit_anything = True
                closest_so_far = temp_rec.t
                rec = temp_rec
                # we return the material info not the material because
                # taichi doesn't yet deal with assigning object pointers
                mat_info = self.spheres[i].material

        return hit_anything, rec, mat_info
