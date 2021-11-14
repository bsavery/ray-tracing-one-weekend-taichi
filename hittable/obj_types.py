from vector import *


MOVING_SPHERE = 1
SPHERE = 0
XY_RECT = 2
XZ_RECT = 3
YZ_RECT = 4


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
    return type_value


def get_bounding_box(obj):
    obj_type = get_object_type(obj)
    if obj_type == SPHERE:
        return (obj.center - Vector(obj.radius)), (obj.center + Vector(obj.radius))
    elif obj_type == XY_RECT:
        return (Point([obj.x0, obj.y0, obj.k-0.0001]), Point([obj.x1, obj.y1, obj.k+0.0001]))
    elif obj_type == XZ_RECT:
        return (Point([obj.x0, obj.k-0.0001, obj.z0]), Point([obj.x1, obj.k+0.0001, obj.z1]))
    elif obj_type == YZ_RECT:
        return (Point([obj.k-0.0001, obj.y0, obj.z0]), Point([obj.k+0.0001, obj.y1, obj.z1]))
    else:
        box0_min, box0_max = (obj.center0 - Vector(obj.radius)), (obj.center0 + Vector(obj.radius))
        box1_min, box1_max = (obj.center1 - Vector(obj.radius)), (obj.center1 + Vector(obj.radius))

        return ti.min(box0_min, box1_min), ti.max(box0_max, box1_max)


def get_center(obj):
    obj_type = get_object_type(obj)
    if obj_type == SPHERE:
        return obj.center
    elif obj_type == XY_RECT:
        cen_x = (obj.x0 + obj.x1) / 2.0
        cen_y = (obj.y0 + obj.y1) / 2.0
        return Point([cen_x, cen_y, obj.k])
    elif obj_type == XZ_RECT:
        cen_x = (obj.x0 + obj.x1) / 2.0
        cen_z = (obj.z0 + obj.z1) / 2.0
        return Point([cen_x, obj.k, cen_z])
    elif obj_type == YZ_RECT:
        cen_y = (obj.y0 + obj.y1) / 2.0
        cen_z = (obj.z0 + obj.z1) / 2.0
        return Point([obj.k, cen_y, cen_z])
    else:
        return (obj.center0 + obj.center1) / 2.0
