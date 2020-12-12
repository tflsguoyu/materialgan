from util import *

class MapsFromPng:
    def __init__(self, fn):
        tex = Image.open(fn)
        tex = gyPIL2Array(tex)
        res = tex.shape[0]
        albedo_np = gyApplyGamma(tex[:, :res, :], 2.2)
        normal_x = tex[:, res:2*res, 0]*2-1
        normal_y = tex[:, res:2*res, 1]*2-1
        normal_xy = (normal_x**2 + normal_y**2).clip(min=0, max=1)
        normal_z  = np.sqrt(1 - normal_xy)
        normal_np = np.stack((normal_x, normal_y, normal_z), 2)
        normal_np = normal_np / np.linalg.norm(normal_np, axis=2, keepdims=True)
        rough_np = gyApplyGamma(tex[:, 2*res:3*res, :], 2.2)
        specular_np = gyApplyGamma(tex[:, 3*res:4*res, :], 2.2)

        self.albedo   = tf.convert_to_tensor(albedo_np,   dtype=tf.float32)
        self.normal   = tf.convert_to_tensor(normal_np,   dtype=tf.float32)
        self.rough    = tf.convert_to_tensor(rough_np,    dtype=tf.float32)
        self.specular = tf.convert_to_tensor(specular_np, dtype=tf.float32)

        self.res = res
        self.N = 1

        self.albedo   = tf.broadcast_to(self.albedo,   [self.N, self.res, self.res, 3])
        self.normal   = tf.broadcast_to(self.normal,   [self.N, self.res, self.res, 3])
        self.rough    = tf.broadcast_to(self.rough,    [self.N, self.res, self.res, 3])
        self.specular = tf.broadcast_to(self.specular, [self.N, self.res, self.res, 3])

class Light:
    def __init__(self, position, intensity, res, N):
        self.position  = tf.convert_to_tensor(position,  dtype=tf.float32)
        self.intensity = tf.convert_to_tensor(intensity, dtype=tf.float32)
        self.position  = tf.broadcast_to(self.position,  [N, res, res, 3])
        self.intensity = tf.broadcast_to(self.intensity, [N, res, res, 3])

class Camera:
    def __init__(self, position, res, N):
        self.position = tf.convert_to_tensor(position, dtype=tf.float32)
        self.position = tf.broadcast_to(self.position, [N, res, res, 3])

class Microfacet:
    def __init__(self, res, N, size):
        self.res = res
        self.N = N
        self.size = size
        self.eps = 1e-6

        self.initGeometry()

    def initGeometry(self):
        tmp = tf.range(0.0, self.res)
        tmp = ((tmp + 0.5) / self.res - 0.5) * self.size
        x, y = tf.meshgrid(tmp, tmp)
        self.pos = tf.stack((x, -y, tf.zeros_like(x)), 2)
        self.pos = tf.broadcast_to(self.pos, [self.N, self.res, self.res, 3])
        self.pos_norm = tf.norm(self.pos, axis=-1, keepdims=True)
        self.pos_norm = tf.concat([self.pos_norm,self.pos_norm,self.pos_norm], axis=-1)

    def GGX(self, cos_h, alpha):
        c2 = cos_h**2
        a2 = alpha**2
        den = c2 * a2 + (1 - c2)
        return a2 / (np.pi * den**2 + self.eps)

    def Fresnel_S(self, cos, specular):
        sphg = tf.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos);
        return specular + (1.0 - specular) * sphg

    def Smith(self, n_dot_v, n_dot_l, alpha):
        def _G1(cos, k):
            return cos / (cos * (1.0 - k) + k)

        k = alpha * 0.5 + self.eps
        return _G1(n_dot_v, k) * _G1(n_dot_l, k)

    def normalize(self, vec):
        vec = vec / tf.norm(vec, axis=-1, keepdims=True)
        return vec

    def square_norm(self, vec):
        vec = tf.norm(vec, axis=-1, keepdims=True)
        vec = vec**2
        vec = tf.concat([vec,vec,vec], axis=-1)
        return vec

    def getDir(self, pos):
        vec = pos - self.pos
        return self.normalize(vec), self.square_norm(vec)

    def dot(self, a, b):
        ab = tf.reduce_sum(a*b, -1, keepdims=True)
        ab = tf.clip_by_value(ab, 0, 999.0)
        ab = tf.concat([ab,ab,ab], axis=-1)
        return ab

    def render(self, map, light, camera):

        v, _ = self.getDir(camera.position)
        l, dist_l_sq = self.getDir(light.position)
        h = self.normalize(l + v)

        n_dot_v = self.dot(map.normal, v)
        n_dot_l = self.dot(map.normal, l)
        n_dot_h = self.dot(map.normal, h)
        v_dot_h = self.dot(v, h)

        geom = n_dot_l / dist_l_sq

        D = self.GGX(n_dot_h, map.rough**2)
        F = self.Fresnel_S(v_dot_h, map.specular)
        G = self.Smith(n_dot_v, n_dot_l, map.rough**2)

        # lambert brdf
        f1 = map.albedo / np.pi
        f1 *= (1 - map.specular)
        # cook-torrence brdf
        f2 = D * F * G / (4 * n_dot_v * n_dot_l + self.eps)

        # brdf
        kd = 1; ks = 1
        f = kd * f1 + ks * f2

        # rendering
        img = f * geom * light.intensity

        return tf.clip_by_value(img, 0.0, 1.0)


def renderTex(fn_tex, res, size, lp, cp, L, fn_im):
    materialObj = MapsFromPng(fn_tex)
    if res > materialObj.res:
        print("[Warning in render.py::renderTex()]: request resolution is larger than texture resolution")
        exit()
    lightObj = Light(lp, L, materialObj.res, materialObj.N)
    cameraObj = Camera(cp, materialObj.res, materialObj.N)

    microfacetObj = Microfacet(materialObj.res, materialObj.N, size)
    im = microfacetObj.render(materialObj, lightObj, cameraObj)
    im = tf.Session().run(im)
    im = gyApplyGamma(im[0,:], 1/2.2)
    im = gyArray2PIL(im)
    if res < materialObj.res:
        im = im.resize((res, res), Image.LANCZOS)
    if fn_im is not None:
        im.save(fn_im)
    return im
