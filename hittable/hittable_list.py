from hittable.bvh import build
import taichi as ti
from vector import *
from .hit_record import empty_hit_record
from .sphere import sphere
from .sphere import hit as hit_sphere
from .moving_sphere import moving_sphere
from .moving_sphere import hit as hit_moving_sphere
from .xy_rect import hit as hit_xy_rect
from .xz_rect import hit as hit_xz_rect
from .yz_rect import hit as hit_yz_rect
from .xy_rect import xy_rect
from .yz_rect import yz_rect
from .xz_rect import xz_rect
from .volume import volume, hit as volume_hit
from .box import box, hit as box_hit
from material import empty_material
from .obj_types import *
from .bvh import build, hit_aabb


@ti.data_oriented
class HittableList:
    def __init__(self):
        self.objects = {}

    def add(self, object):
        # set the id and add to list
        obj_type = get_object_type(object)
        if obj_type in self.objects.keys():
            object.id = len(self.objects[obj_type])
            self.objects[obj_type].append(object)
        else:
            object.id = 0
            self.objects[obj_type] = [object]

    def commit(self):
        self.bvh = build(self.objects)

        def fill_array(from_array, to_array):
            for i, obj in enumerate(from_array):
                to_array[i] = obj

        struct_types = {SPHERE: ('sphere', sphere), MOVING_SPHERE: ('moving_sphere', moving_sphere),
                        XY_RECT: ('xy_rect', xy_rect), YZ_RECT: ('yz_rect', yz_rect),
                        XZ_RECT: ('xz_rect', xz_rect), BOX: ('box', box),
                        VOLUME: ('volume', volume)}
        for obj_type in [SPHERE, MOVING_SPHERE, XY_RECT, YZ_RECT, XZ_RECT, BOX, VOLUME]:
            struct_name, struct_type = struct_types[obj_type]
            if obj_type in self.objects.keys():
                num = len(self.objects[obj_type])
                struct_field = struct_type.field(shape=num)
                fill_array(self.objects[obj_type], struct_field)
            else:
                struct_field = struct_type.field(shape=1)

            setattr(self, struct_name, struct_field)

    @ti.func
    def hit(self, r, t_min, t_max):
        hit_anything = False
        closest_so_far = t_max
        rec = empty_hit_record()
        mat_info = empty_material()

        curr = 0

        # walk the bvh tree
        while curr != -1:
            bvh_node = self.bvh[curr]

            if bvh_node.obj_id != -1:
                # this is a leaf node, check the sphere
                hit, temp_rec, temp_mat_info = self.hit_obj(bvh_node.obj_type, bvh_node.obj_id, r,
                                                            t_min, closest_so_far)
                if hit:
                    hit_anything = True
                    closest_so_far = temp_rec.t
                    rec = temp_rec
                    mat_info = temp_mat_info
                curr = bvh_node.next_id
            else:
                if hit_aabb(bvh_node, r, t_min, closest_so_far):
                    # visit left child next (left child will visit it's next = right)
                    curr = bvh_node.left_id
                else:
                    curr = bvh_node.next_id

        return hit_anything, rec, mat_info

    @ti.func
    def hit_obj(self, obj_type, obj_id, r, t_min, t_max):
        hit = False
        rec = empty_hit_record()
        mat = empty_material()

        if obj_type == MOVING_SPHERE:
            hit, rec, mat = hit_moving_sphere(self.moving_sphere[obj_id], r, t_min, t_max)
        elif obj_type == SPHERE:
            hit, rec, mat = hit_sphere(self.sphere[obj_id], r, t_min, t_max)
        elif obj_type == XY_RECT:
            hit, rec, mat = hit_xy_rect(self.xy_rect[obj_id], r, t_min, t_max)
        elif obj_type == YZ_RECT:
            hit, rec, mat = hit_yz_rect(self.yz_rect[obj_id], r, t_min, t_max)
        elif obj_type == XZ_RECT:
            hit, rec, mat = hit_xz_rect(self.xz_rect[obj_id], r, t_min, t_max)
        elif obj_type == BOX:
            hit, rec, mat = box_hit(self.box[obj_id], r, t_min, t_max)
        elif obj_type == VOLUME:
            hit, rec, mat = volume_hit(self.volume[obj_id], r, t_min, t_max)
        return hit, rec, mat
