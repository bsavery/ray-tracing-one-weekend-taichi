import taichi as ti
import copy
import random
from vector import *
from .obj_types import *


# struct for bvh node
# holds bvh box bounds and either the object reference or ids of other bvh nodes
# for left, right, next in tree
BVHNode = ti.types.struct(box_min=Point, box_max=Point,
                          obj_type=ti.i32, obj_id=ti.i32,
                          left_id=ti.i32, right_id=ti.i32,
                          parent_id=ti.i32, next_id=ti.i32)


def surrounding_box(box1, box2):
    ''' Calculates the surround bbox of two bboxes '''
    box1_min, box1_max = box1
    box2_min, box2_max = box2

    small = ti.min(box1_min, box2_min)
    big = ti.max(box1_max, box2_max)
    return small, big


def sort_obj_list(obj_list):
    ''' Sort the list of objects along the longest directional span '''
    def get_x(e):
        obj = e
        return get_center(obj).x

    def get_y(e):
        obj = e
        return get_center(obj).y

    def get_z(e):
        obj = e
        return get_center(obj).z

    centers = [get_center(obj) for obj in obj_list]
    min_center = [
        min([center[0] for center in centers]),
        min([center[1] for center in centers]),
        min([center[2] for center in centers])
    ]
    max_center = [
        max([center[0] for center in centers]),
        max([center[1] for center in centers]),
        max([center[2] for center in centers])
    ]
    span_x, span_y, span_z = (max_center[0] - min_center[0],
                              max_center[1] - min_center[1],
                              max_center[2] - min_center[2])
    if span_x >= span_y and span_x >= span_z:
        obj_list.sort(key=get_x)
    elif span_y >= span_z:
        obj_list.sort(key=get_y)
    else:
        obj_list.sort(key=get_z)
    return obj_list


def build_bvh(object_list, original_list, parent_id=-1, curr_id=0):
    obj_list = copy.copy(object_list)
    node_list = []

    span = len(object_list)
    if span == 1:
        # one obj, set to obj bbox
        obj = object_list[0]
        obj_type = get_object_type(obj)
        bbox_min, bbox_max = get_bounding_box(obj)
        node_list.append(
            BVHNode(box_min=bbox_min, box_max=bbox_max,
                    obj_type=obj_type, obj_id=obj.id,
                    left_id=-1, right_id=-1,
                    parent_id=parent_id, next_id=-1))

    else:
        # sort list of object and divide and conquer
        sorted_list = sort_obj_list(obj_list)
        mid = int(span / 2)

        # pass correct start indices to sublists generation
        left_nodelist = build_bvh(sorted_list[:mid], original_list, curr_id, curr_id + 1)
        right_nodelist = build_bvh(sorted_list[mid:], original_list,
                                   curr_id, curr_id + len(left_nodelist) + 1)

        box_min, box_max = surrounding_box(
            (left_nodelist[0].box_min, left_nodelist[0].box_max),
            (right_nodelist[0].box_min, right_nodelist[0].box_max))

        # create this node
        node_list.append(
            BVHNode(box_min=box_min, box_max=box_max,
                    obj_type=-1, obj_id=-1,
                    left_id=curr_id + 1, right_id=curr_id + len(left_nodelist) + 1,
                    parent_id=parent_id, next_id=-1))
        # add right and left
        node_list = node_list + left_nodelist + right_nodelist

    return node_list


def set_next_id_links(bvh_node_list):
    ''' given a list of nodes set the 'next_id' link in the nodes '''
    def inner_loop(node_id):
        node = bvh_node_list[node_id]
        if node.parent_id == -1:
            return -1

        parent = bvh_node_list[node.parent_id]
        if parent.right_id != -1 and parent.right_id != node_id:
            return parent.right_id
        else:
            return inner_loop(node.parent_id)

    for i, node in enumerate(bvh_node_list):
        node.next_id = inner_loop(i)


def build(obj_list):
    ''' building function. Compress the object list to structure'''

    # construct temp list of node structs
    total_list = []
    for li in obj_list.values():
        total_list += li
    bvh_node_list = build_bvh(total_list, obj_list)
    set_next_id_links(bvh_node_list)
    bvh_field = BVHNode.field(shape=(len(bvh_node_list),))

    for i, node in enumerate(bvh_node_list):
        bvh_field[i] = node
    return bvh_field


@ti.func
def hit_aabb(bvh_node, r, t_min, t_max):
    intersect = True
    min_aabb, max_aabb = bvh_node.box_min, bvh_node.box_max
    ray_direction, ray_origin = r.dir, r.orig

    for i in ti.static(range(3)):
        if ray_direction[i] == 0:
            if ray_origin[i] < min_aabb[i] or ray_origin[i] > max_aabb[i]:
                intersect = False
        else:
            i1 = (min_aabb[i] - ray_origin[i]) / ray_direction[i]
            i2 = (max_aabb[i] - ray_origin[i]) / ray_direction[i]

            new_t_max = ti.max(i1, i2)
            new_t_min = ti.min(i1, i2)

            t_max = ti.min(new_t_max, t_max)
            t_min = ti.max(new_t_min, t_min)

    if t_min > t_max:
        intersect = False
    return intersect
