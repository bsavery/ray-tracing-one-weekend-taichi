import taichi as ti


@ti.func
def at(origin, direction, t):
    return origin + direction * t


@ti.data_oriented
class Rays:
    ''' An array of "in flight" rays'''
    def __init__(self, x, y):
        self.origin = ti.Vector.field(3, dtype=ti.f32)
        self.direction = ti.Vector.field(3, dtype=ti.f32)
        self.depth = ti.field(ti.i32)
        self.attenuation = ti.Vector.field(3, dtype=ti.f32)
        ti.root.dense(ti.ij, (x, y)).place(self.origin, self.direction,
                                           self.depth, self.attenuation)

    @ti.func
    def set(self, x, y, ray_org, ray_dir, depth, attenuation):
        self.origin[x, y] = ray_org
        self.direction[x, y] = ray_dir
        self.depth[x, y] = depth
        self.attenuation[x, y] = attenuation

    @ti.func
    def get(self, x, y):
        return self.origin[x, y], self.direction[x, y], self.depth[
            x, y], self.attenuation[x, y]

    @ti.func
    def get_od(self, x, y):
        return self.origin[x, y], self.direction[x, y]

    @ti.func
    def get_depth(self, x, y):
        return self.depth[x, y]

    @ti.func
    def set_depth(self, x, y, d):
        self.depth[x, y] = d


@ti.data_oriented
class HitRecord:
    def __init__(self, x, y):
        self.hit = ti.field(ti.i32)
        self.p = ti.Vector.field(3, dtype=ti.f32)
        self.n = ti.Vector.field(3, dtype=ti.f32)
        self.front_facing = ti.field(ti.i32)
        self.mat_index = ti.field(ti.i32)
        ti.root.dense(ti.ij, (x, y)).place(self.hit, self.p, self.n,
                                           self.front_facing, self.mat_index)

    @ti.func
    def set(self, x, y, hit, p, n, front_facing, mat_index):
        self.hit[x, y] = hit
        self.p[x, y] = p
        self.n[x, y] = n
        self.front_facing[x, y] = front_facing
        self.mat_index[x, y] = mat_index

    @ti.func
    def get(self, x, y):
        return self.hit[x, y], self.p[x, y], self.n[x, y], self.front_facing[
            x, y], self.mat_index[x, y]

    @ti.func
    def get_hit(self, x, y):
        return self.hit[x, y]

    @ti.func
    def set_hit(self, x, y, hit):
        self.hit[x, y] = hit
