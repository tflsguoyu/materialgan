from util import *

def tex2map(tex):
    albedo = ((tex[:,0:3,:,:].clamp(-1,1) + 1) / 2) ** 2.2

    normal_x  = tex[:,3,:,:].clamp(-1,1)
    normal_y  = tex[:,4,:,:].clamp(-1,1)
    normal_xy = (normal_x**2 + normal_y**2).clamp(min=0, max=1-eps)
    normal_z  = (1 - normal_xy).sqrt()
    normal    = th.stack((normal_x, normal_y, normal_z), 1)
    normal    = normal.div(normal.norm(2.0, 1, keepdim=True))

    rough = ((tex[:,5,:,:].clamp(-0.3,1) + 1) / 2) ** 2.2
    rough = rough.clamp(min=eps).unsqueeze(1).expand(-1,3,-1,-1)

    if tex.shape[1] == 9:
        specular = ((tex[:,6:9,:,:].clamp(-1,1) + 1) / 2) ** 2.2
        return albedo, normal, rough, specular

    return albedo, normal, rough

class Microfacet:
    def __init__(self, res, size, f0=0.04):
        self.res = res
        self.size = size
        self.f0 = f0
        self.eps = 1e-6

        self.initGeometry()

    def initGeometry(self):
        tmp = th.arange(self.res, dtype=th.float32).cuda()
        tmp = ((tmp + 0.5) / self.res - 0.5) * self.size
        y, x = th.meshgrid((tmp, tmp))
        self.pos = th.stack((x, -y, th.zeros_like(x)), 2)
        self.pos_norm = self.pos.norm(2.0, 2, keepdim=True)

    def GGX(self, cos_h, alpha):
        c2 = cos_h ** 2
        a2 = alpha ** 2
        den = c2 * a2 + (1 - c2)
        return a2 / (np.pi * den**2 + self.eps)

    def Beckmann(self, cos_h, alpha):
        c2 = cos_h ** 2
        t2 = (1 - c2) / c2
        a2 = alpha ** 2
        return th.exp(-t2 / a2) / (np.pi * a2 * c2 ** 2)

    def Fresnel(self, cos, f0):
        return f0 + (1 - f0) * (1 - cos)**5

    def Fresnel_S(self, cos, specular):
        sphg = th.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos);
        return specular + (1.0 - specular) * sphg

    def Smith(self, n_dot_v, n_dot_l, alpha):
        def _G1(cos, k):
            return cos / (cos * (1.0 - k) + k)

        k = alpha * 0.5 + self.eps
        return _G1(n_dot_v, k) * _G1(n_dot_l, k)

    def normalize(self, vec):
        assert(vec.size(0)==self.N)
        assert(vec.size(1)==3)
        assert(vec.size(2)==self.res)
        assert(vec.size(3)==self.res)

        vec = vec / (vec.norm(2.0, 1, keepdim=True))
        return vec

    def getDir(self, pos):
        pos = th.from_numpy(pos).cuda()
        vec = (pos - self.pos).permute(2,0,1).unsqueeze(0).expand(self.N,-1,-1,-1)
        return self.normalize(vec), (vec**2).sum(1, keepdim=True).expand(-1,3,-1,-1)

    def AdotB(self, a, b):
        ab = (a*b).sum(1, keepdim=True).clamp(min=0).expand(-1,3,-1,-1)
        return ab

    def eval(self, textures, lightPos, cameraPos, light):
        self.N = textures.size(0)
        isSpecular = False
        if textures.size(1) == 9:
            isSpecular = True
            # print('Render Specular\n')

        if isSpecular:
            albedo, normal, rough, specular = tex2map(textures)
        else:
            albedo, normal, rough = tex2map(textures)
        light = light.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(albedo)

        v, _ = self.getDir(cameraPos)
        l, dist_l_sq = self.getDir(lightPos)
        h = self.normalize(l + v)

        n_dot_v = self.AdotB(normal, v)
        n_dot_l = self.AdotB(normal, l)
        n_dot_h = self.AdotB(normal, h)
        v_dot_h = self.AdotB(v, h)

        geom = n_dot_l / dist_l_sq

        D = self.GGX(n_dot_h, rough**2)
        # D = self.Beckmann(n_dot_h, rough**2)
        if isSpecular:
            F = self.Fresnel_S(v_dot_h, specular)
        else:
            F = self.Fresnel(v_dot_h, self.f0)
        G = self.Smith(n_dot_v, n_dot_l, rough**2)

        # lambert brdf
        f1 = albedo / np.pi
        if isSpecular:
            f1 *= (1 - specular)
        # cook-torrence brdf
        f2 = D * F * G / (4 * n_dot_v * n_dot_l + self.eps)

        # brdf
        kd = 1; ks = 1
        f = kd * f1 + ks * f2

        # rendering
        img = f * geom * light

        return img.clamp(0,1)

def png2tex(fn):
    png = Image.open(fn)
    png = gyPIL2Array(png)

    res = png.shape[0]
    isSpecular = False
    if png.shape[1]//png.shape[0] == 4:
        isSpecular = True
    albedo = png[:,:res,:]
    normal = png[:,res:res*2,0:2]
    rough = png[:,res*2:res*3,0]
    tex = th.cat((th.from_numpy(albedo), th.from_numpy(normal), th.from_numpy(rough).unsqueeze(2)), 2)
    if isSpecular:
        # print('Specular!!')
        specular = png[:,res*3:res*4,:]
        tex = th.cat((tex,th.from_numpy(specular)),2)
    tex = tex * 2 - 1
    return tex.permute(2,0,1).unsqueeze(0).cuda(), res

def tex2png(tex, fn, isVertical=False):
    isSpecular = False
    if tex.size(1) == 9:
        isSpecular = True

    if isSpecular:
        albedo, normal, rough, specular = tex2map(tex)
    else:
        albedo, normal, rough = tex2map(tex)

    albedo = gyTensor2Array(albedo[0,:].permute(1,2,0))
    normal = gyTensor2Array((normal[0,:].permute(1,2,0)+1)/2)
    rough  = gyTensor2Array(rough[0,:].permute(1,2,0))

    albedo = gyArray2PIL(gyApplyGamma(albedo, 1/2.2))
    normal = gyArray2PIL(normal)
    rough  = gyArray2PIL(gyApplyGamma(rough, 1/2.2))

    if isVertical:
        png = gyConcatPIL_v(gyConcatPIL_v(albedo,normal), rough)
    else:
        png = gyConcatPIL_h(gyConcatPIL_h(albedo,normal), rough)

    if isSpecular:
        specular = gyTensor2Array(specular[0,:].permute(1,2,0))
        specular = gyArray2PIL(gyApplyGamma(specular, 1/2.2))
        if isVertical:
            png = gyConcatPIL_v(png, specular)
        else:
            png = gyConcatPIL_h(png, specular)

    if fn is not None:
        png.save(fn)
    return png

def renderTex(fn_tex, res, size, lp, cp, L, fn_im):
    # print(lp, cp, L)
    textures, tex_res = png2tex(fn_tex)
    # tex2png(textures, 'a.png')
    # exit()
    if res > tex_res:
        print("[Warning in render.py::renderTex()]: request resolution is larger than texture resolution")
        exit()
    renderObj = Microfacet(res=tex_res, size=size)
    im = renderObj.eval(textures, lightPos=lp, \
        cameraPos=cp, light=th.from_numpy(L).cuda())
    im = gyApplyGamma(gyTensor2Array(im[0,:].permute(1,2,0)), 1/2.2)
    im = gyArray2PIL(im)
    if res < tex_res:
        im = im.resize((res, res), Image.LANCZOS)
    if fn_im is not None:
        im.save(fn_im)
    return im