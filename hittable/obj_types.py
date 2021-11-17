from vector import *


MOVING_SPHERE = 1
SPHERE = 0
XY_RECT = 2
XZ_RECT = 3
YZ_RECT = 4
BOX = 5


def get_object_type(obj):
    if 'x0' in obj.keys and 'y0' in obj.keys:
        type_value = XY_RECT
    elif 'x0' in obj.keys and 'z0' in obj.keys:
        type_value = XZ_RECT
    elif 'y0' in obj.keys and 'z0' in obj.keys:
        type_value = YZ_RECT
    elif 'center0' in obj.keys:
        type_value = MOVING_SPHERE
    elif 'center' in obj.keys:
        type_value = SPHERE
    elif 'left' in obj.keys:
        type_value = BOX
    return type_value


def get_bounding_box(obj):
    return obj.bbox_min, obj.bbox_max


def get_center(obj):
    box_min, box_max = get_bounding_box(obj)
    return (box_min + box_max) / 2.0
