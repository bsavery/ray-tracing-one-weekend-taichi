from vector import *


MOVING_SPHERE = 1
SPHERE = 0


def get_object_type(obj):
    return SPHERE if 'center' in obj.keys else MOVING_SPHERE


def get_bounding_box(obj):
    if get_object_type(obj) == SPHERE:
        return (obj.center - Vector(obj.radius)), (obj.center + Vector(obj.radius))
    else:
        box0_min, box0_max = (obj.center0 - Vector(obj.radius)), (obj.center0 + Vector(obj.radius))
        box1_min, box1_max = (obj.center1 - Vector(obj.radius)), (obj.center1 + Vector(obj.radius))

        return ti.min(box0_min, box1_min), ti.max(box0_max, box1_max)


def get_center(obj):
    if 'center' in obj.keys:
        return obj.center
    else:
        return (obj.center0 + obj.center1) / 2.0
