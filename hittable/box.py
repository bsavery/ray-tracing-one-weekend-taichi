import taichi as ti
from taichi.testing import test
from vector import *
from . import empty_hit_record, set_face_normal
import ray
from .xy_rect import xy_rect, XYRect, hit as xy_hit
from .yz_rect import yz_rect, YZRect, hit as yz_hit
from .xz_rect import xz_rect, XZRect, hit as xz_hit
from material import Material
import math
import sys

# struct for sphere
box = ti.types.struct(front=xy_rect, back=xy_rect,
                      left=yz_rect, right=yz_rect,
                      top=xz_rect, bottom=xz_rect,
                      offset=Point, sin_theta=ti.f32, cos_theta=ti.f32,
                      bbox_min=Point, bbox_max=Point)


def Box(p0, p1, offset, rotation, material):

    radians = math.radians(rotation)
    sin_theta = math.sin(radians)
    cos_theta = math.cos(radians)
    bbox_min, bbox_max = p0, p1

    min_val = Point([sys.float_info.max] * 3)
    max_val = Point([-sys.float_info.max] * 3)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = i * bbox_max + (1-i) * bbox_min

                newx = cos_theta * val.x + sin_theta * val.z
                newz = -sin_theta * val.x + cos_theta * val.z

                tester = Point([newx, val.y, newz])

                min_val = ti.min(min_val, tester)
                max_val = ti.max(max_val, tester)

    return box(front=XYRect(p0.x, p1.x, p0.y, p1.y, p1.z, material),
               back=XYRect(p0.x, p1.x, p0.y, p1.y, p0.z, material),
               left=YZRect(p0.y, p1.y, p0.z, p1.z, p1.x, material),
               right=YZRect(p0.y, p1.y, p0.z, p1.z, p0.x, material),
               top=XZRect(p0.x, p1.x, p0.z, p1.z, p1.y, material),
               bottom=XZRect(p0.x, p1.x, p0.z, p1.z, p0.y, material),
               offset=offset, sin_theta=sin_theta, cos_theta=cos_theta,
               bbox_min=(min_val + offset), bbox_max=(max_val+offset))


@ti.func
def hit(box, r, t_min, t_max):
    ''' Intersect a ray with a given center and radius.
        Note we pass in the hit record by reference. '''
    hit = False
    rec = empty_hit_record()

    r.orig -= box.offset
    orig_x = r.orig.x
    r.orig.x = box.cos_theta*r.orig.x - box.sin_theta*r.orig.z
    r.orig.z = box.sin_theta*orig_x + box.cos_theta*r.orig.z

    dir_x = r.dir.x
    r.dir.x = box.cos_theta*r.dir.x - box.sin_theta*r.dir.z
    r.dir.z = box.sin_theta*dir_x + box.cos_theta*r.dir.z

    temp_hit, temp_rec, _ = xy_hit(box.front, r, t_min, t_max)
    if temp_hit:
        hit = temp_hit
        rec = temp_rec
        t_max = temp_rec.t

    temp_hit, temp_rec, _ = xy_hit(box.back, r, t_min, t_max)
    if temp_hit:
        hit = temp_hit
        rec = temp_rec
        t_max = temp_rec.t

    temp_hit, temp_rec, _ = yz_hit(box.left, r, t_min, t_max)
    if temp_hit:
        hit = temp_hit
        rec = temp_rec
        t_max = temp_rec.t

    temp_hit, temp_rec, _ = yz_hit(box.right, r, t_min, t_max)
    if temp_hit:
        hit = temp_hit
        rec = temp_rec
        t_max = temp_rec.t

    temp_hit, temp_rec, _ = xz_hit(box.top, r, t_min, t_max)
    if temp_hit:
        hit = temp_hit
        rec = temp_rec
        t_max = temp_rec.t

    temp_hit, temp_rec, _ = xz_hit(box.bottom, r, t_min, t_max)
    if temp_hit:
        hit = temp_hit
        rec = temp_rec
        t_max = temp_rec.t

    if hit:
        p_x = rec.p.x
        rec.p.x = box.cos_theta*rec.p.x + box.sin_theta*rec.p.z
        rec.p.z = -box.sin_theta*p_x + box.cos_theta*rec.p.z

        norm_x = rec.normal.x
        rec.normal.x = box.cos_theta*rec.normal.x + box.sin_theta*rec.normal.z
        rec.normal.z = -box.sin_theta*norm_x + box.cos_theta*rec.normal.z

        rec.p += box.offset
        set_face_normal(r, rec.normal, rec)

    return hit, rec, box.left.material
