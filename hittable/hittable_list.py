import taichi as ti
from vector import *
from .hit_record import empty_hit_record
from .sphere import sphere
from .sphere import hit as hit_sphere
from .moving_sphere import moving_sphere
from .moving_sphere import hit as hit_moving_sphere
from material import empty_material


@ti.data_oriented
class HittableList:
    def __init__(self):
        self.objects = {'static': [], 'moving': []}

    def add(self, object):
        # sort by checking for key "center"
        if 'center' in object.keys:
            self.objects['static'].append(object)
        else:
            self.objects['moving'].append(object)

    
    
    
    def commit(self):
        ''' Save the sphere data so we can loop over these.'''
        self.n_static = len(self.objects['static'])
        self.static_spheres = sphere.field(shape=(self.n_static,))

        self.n_moving = len(self.objects['moving'])
        self.moving_spheres = moving_sphere.field(shape=(self.n_moving,))

        def fill_array(from_array, to_array):
            for i, obj in enumerate(from_array):
                to_array[i] = obj

        fill_array(self.objects['static'], self.static_spheres)
        fill_array(self.objects['moving'], self.moving_spheres)

    @ti.func
    def hit(self, r, t_min, t_max):
        hit_anything = False
        closest_so_far = t_max
        rec = empty_hit_record()
        mat_info = empty_material()

        for i in range(self.n_static):
            hit, temp_rec = hit_sphere(self.static_spheres[i], r, t_min, closest_so_far)
            if hit:
                hit_anything = True
                closest_so_far = temp_rec.t
                rec = temp_rec
                # we return the material info not the material because
                # taichi doesn't yet deal with assigning object pointers
                mat_info = self.static_spheres[i].material

        for i in range(self.n_moving):
            hit, temp_rec = hit_moving_sphere(self.moving_spheres[i], r, t_min, closest_so_far)
            if hit:
                hit_anything = True
                closest_so_far = temp_rec.t
                rec = temp_rec
                # we return the material info not the material because
                # taichi doesn't yet deal with assigning object pointers
                mat_info = self.moving_spheres[i].material

        return hit_anything, rec, mat_info
