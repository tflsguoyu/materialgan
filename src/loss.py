from util import *
from render import *
from descriptor import FeatureLoss, StyleLoss
from torchvision.transforms import Normalize
sys.path.insert(1, 'PerceptualSimilarity/')
import models
import kornia

np.set_printoptions(precision=4, suppress=True)

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

def plotAndSave(loss, save_dir):
    plt.figure(figsize=(8,4))
    for i in range(loss.shape[1]):
        plt.plot(loss[:,i], label='%.4f' % loss[-1,i])
    plt.legend()
    plt.savefig(save_dir)
    plt.close()

    plt.figure(figsize=(8,4))
    for i in range(loss.shape[1]):
        plt.plot(np.log1p(loss[:,i]), label='%.4f' % np.log1p(loss[-1,i]))
    plt.legend()
    plt.savefig(save_dir[:-4]+'_log.png')
    plt.close()

def normalize_vgg19(input, isGram):
    if isGram:
        transform = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.255]
        )
    else:
        transform = Normalize(
            mean=[0.48501961, 0.45795686, 0.40760392],
            std=[1./255, 1./255, 1./255]
        )
    return transform(input)

class Losses:
    def __init__(self, args, textures_init, textures_ref, rendered_ref, res, size, lp, cp):
        self.args = args
        self.textures_init = textures_init
        self.textures_ref = textures_ref
        self.rendered_ref = rendered_ref
        self.res = res
        self.size = size
        self.lp = lp
        self.cp = cp
        self.criterion = th.nn.MSELoss().to(device)

        self.precompute()

    def precompute(self):

        self.FL_w = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_w)
        for p in self.FL_w.parameters():
            p.requires_grad = False

        self.FL_n = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_n)
        for p in self.FL_n.parameters():
            p.requires_grad = False

        self.FL_wn = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_wn)
        for p in self.FL_wn.parameters():
            p.requires_grad = False

        self.SL = StyleLoss()
        for p in self.SL.parameters():
            p.requires_grad = False

        if device == "cuda:0":
            self.LPIPS = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
        else:
            self.LPIPS = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=False)
        self.Laplacian = kornia.filters.Laplacian(3, normalized=False)

        if self.args.embed_tex:
            self.albedo_ref_pix, self.normal_ref_pix, self.rough_ref_pix, self.specular_ref_pix, \
            self.albedo_ref_vgg, self.normal_ref_vgg, self.rough_ref_vgg, self.specular_ref_vgg = \
            self.eval_texture_vgg(self.textures_ref)

            self.fl_albedo_ref_w   = self.FL_w(self.albedo_ref_vgg)
            self.fl_normal_ref_w   = self.FL_w(self.normal_ref_vgg)
            self.fl_rough_ref_w    = self.FL_w(self.rough_ref_vgg)
            self.fl_specular_ref_w = self.FL_w(self.specular_ref_vgg)

            self.fl_albedo_ref_n   = self.FL_n(self.albedo_ref_vgg)
            self.fl_normal_ref_n   = self.FL_n(self.normal_ref_vgg)
            self.fl_rough_ref_n    = self.FL_n(self.rough_ref_vgg)
            self.fl_specular_ref_n = self.FL_n(self.specular_ref_vgg)

            self.fl_albedo_ref_wn   = self.FL_wn(self.albedo_ref_vgg)
            self.fl_normal_ref_wn   = self.FL_wn(self.normal_ref_vgg)
            self.fl_rough_ref_wn    = self.FL_wn(self.rough_ref_vgg)
            self.fl_specular_ref_wn = self.FL_wn(self.specular_ref_vgg)

        else:
            self.rendered_ref_pix = self.rendered_ref
            self.rendered_ref_vgg = self.eval_render_vgg(self.rendered_ref, self.args.jittering)

            if self.args.jittering:
                self.sl_rendered_ref = self.SL(self.rendered_ref_vgg)
            else:
                self.fl_rendered_ref_w = self.FL_w(self.rendered_ref_vgg)
                self.fl_rendered_ref_n = self.FL_n(self.rendered_ref_vgg)
                self.fl_rendered_ref_wn = self.FL_wn(self.rendered_ref_vgg)

    def eval_texture_vgg(self, textures):
        albedo, normal, rough, specular = tex2map(textures)
        albedo    = albedo.clamp(eps,1) ** (1/2.2)
        normal    = (normal+1)/2
        rough     = rough.clamp(eps,1) ** (1/2.2)
        specular  = specular.clamp(eps,1) ** (1/2.2)

        albedo_vgg = normalize_vgg19(albedo[0,:], False).unsqueeze(0)
        normal_vgg = normalize_vgg19(normal[0,:], False).unsqueeze(0)
        rough_vgg  = normalize_vgg19(rough[0,:], False).unsqueeze(0)
        specular_vgg  = normalize_vgg19(specular[0,:], False).unsqueeze(0)

        return albedo, normal, rough, specular, albedo_vgg, normal_vgg, rough_vgg, specular_vgg

    def eval_render_vgg(self, rendered, isGram):
        rendered_tmp = rendered.clone()
        for i in range(self.args.num_render_used):
            rendered_tmp[i,:] = normalize_vgg19(rendered[i,:], isGram)
        return rendered_tmp

    def eval_render_jitter(self, textures, li):

        renderOBJ = Microfacet(res=self.res, size=self.size)
        rendered = th.zeros(self.args.num_render_used, 3, self.res, self.res)
        for i in range(self.args.num_render_used):
            lp_this = self.lp[i,:]
            rendered[i,:] = renderOBJ.eval(textures, lightPos=lp_this, cameraPos=self.cp[i,:], light=th.from_numpy(li).to(device))

        renderOBJ_jitter = Microfacet(res=self.res, size=self.size)
        rendered_jitter = th.zeros(self.args.num_render_used, 3, self.res, self.res).to(device)
        for i in range(self.args.num_render_used):
            lp_this = self.lp[i,:] + np.random.randn(*self.lp[i,:].shape) * 0.1
            rendered_jitter[i,:] = renderOBJ_jitter.eval(textures, lightPos=lp_this, cameraPos=self.cp[i,:], light=th.from_numpy(li).to(device))

        return rendered.clamp(eps,1)**(1/2.2), rendered_jitter.clamp(eps,1)**(1/2.2)

    def eval_render(self, textures, li):

        renderOBJ = Microfacet(res=self.res, size=self.size)
        rendered = th.zeros(self.args.num_render_used, 3, self.res, self.res).to(device)
        for i in range(self.args.num_render_used):
            lp_this = self.lp[i,:]
            rendered[i,:] = renderOBJ.eval(textures, lightPos=lp_this, cameraPos=self.cp[i,:], light=th.from_numpy(li).to(device))
        return rendered.clamp(eps,1)**(1/2.2)


    def eval(self, textures_pre, light, type, epoch):
        loss = 0
        losses = np.array([0,0,0,0]).astype(np.float32)

        if self.args.embed_tex:
            albedo_pre_pix, normal_pre_pix, rough_pre_pix, specular_pre_pix, \
            albedo_pre_vgg, normal_pre_vgg, rough_pre_vgg, specular_pre_vgg = \
            self.eval_texture_vgg(textures_pre)

            if self.args.loss_weight[0] > -eps:
                # pixel loss
                loss_pl  = self.criterion(albedo_pre_pix, self.albedo_ref_pix)
                loss_pl += self.criterion(normal_pre_pix, self.normal_ref_pix)
                loss_pl += self.criterion(rough_pre_pix,  self.rough_ref_pix)
                loss_pl += self.criterion(specular_pre_pix,  self.specular_ref_pix)

                if self.args.loss_weight[0] > eps:
                    loss_pl *= self.args.loss_weight[0]
                    loss += loss_pl
                losses[0] = loss_pl.item()


            if self.args.loss_weight[1] > -eps:
                # feature loss
                if type == 'L':
                    loss_fl  = self.criterion(self.FL_w(albedo_pre_vgg), self.fl_albedo_ref_w) * 0.1
                    loss_fl += self.criterion(self.FL_w(normal_pre_vgg), self.fl_normal_ref_w) * 0.7
                    loss_fl += self.criterion(self.FL_w(rough_pre_vgg),  self.fl_rough_ref_w) * 0.1
                    loss_fl += self.criterion(self.FL_w(specular_pre_vgg),  self.fl_specular_ref_w) * 0.1
                elif type == 'N':
                    loss_fl  = self.criterion(self.FL_n(albedo_pre_vgg), self.fl_albedo_ref_n) * 0.1
                    loss_fl += self.criterion(self.FL_n(normal_pre_vgg), self.fl_normal_ref_n) * 0.7
                    loss_fl += self.criterion(self.FL_n(rough_pre_vgg),  self.fl_rough_ref_n) * 0.1
                    loss_fl += self.criterion(self.FL_n(specular_pre_vgg),  self.fl_specular_ref_n) * 0.1
                elif type == 'LN':
                    loss_fl  = self.criterion(self.FL_wn(albedo_pre_vgg), self.fl_albedo_ref_wn) * 0.1
                    loss_fl += self.criterion(self.FL_wn(normal_pre_vgg), self.fl_normal_ref_wn) * 0.7
                    loss_fl += self.criterion(self.FL_wn(rough_pre_vgg),  self.fl_rough_ref_wn) * 0.1
                    loss_fl += self.criterion(self.FL_wn(specular_pre_vgg),  self.fl_specular_ref_wn) * 0.1
                else:
                    print('Latent type wrong!')
                    exit()

                if self.args.loss_weight[1] > eps:
                    loss_fl *= self.args.loss_weight[1]
                    loss += loss_fl
                losses[1] = loss_fl.item()

            if self.args.loss_weight[2] > -eps:
                # pixel loss
                loss_lpips  = self.LPIPS.forward(albedo_pre_pix, self.albedo_ref_pix).sum() * 0.1
                loss_lpips += self.LPIPS.forward(normal_pre_pix, self.normal_ref_pix).sum() * 0.7
                loss_lpips += self.LPIPS.forward(rough_pre_pix,  self.rough_ref_pix).sum() * 0.1
                loss_lpips += self.LPIPS.forward(specular_pre_pix,  self.specular_ref_pix).sum() * 0.1

                if self.args.loss_weight[2] > eps:
                    loss_lpips *= self.args.loss_weight[2]
                    loss += loss_lpips
                losses[2] = loss_lpips.item()

        else:
            if self.args.jittering:
                rendered_pre_pix, rendered_pre_pix_jitter = self.eval_render_jitter(textures_pre, light)
                rendered_pre_vgg = self.eval_render_vgg(rendered_pre_pix_jitter,True)
            else:
                rendered_pre_pix = self.eval_render(textures_pre, light)
                rendered_pre_vgg = self.eval_render_vgg(rendered_pre_pix, False)

            if self.args.loss_weight[0] > -eps:
                # pixel loss
                loss_pl = self.criterion(rendered_pre_pix, self.rendered_ref_pix)
                # print('loss_pl:', loss_pl)
                if self.args.loss_weight[0] > eps:
                    loss_pl *= self.args.loss_weight[0]
                    loss += loss_pl
                losses[0] = loss_pl.item()

            if self.args.loss_weight[1] > -eps:
                # feature loss
                if self.args.jittering:
                    loss_fl = self.criterion(self.SL(rendered_pre_vgg), self.sl_rendered_ref)
                else:
                    if type == 'L':
                        loss_fl = self.criterion(self.FL_w(rendered_pre_vgg), self.fl_rendered_ref_w)
                    elif type == 'N':
                        loss_fl = self.criterion(self.FL_n(rendered_pre_vgg), self.fl_rendered_ref_n)
                    elif type == 'LN':
                        loss_fl = self.criterion(self.FL_wn(rendered_pre_vgg), self.fl_rendered_ref_wn)
                    else:
                        print('Latent type wrong!')
                        exit()

                if self.args.loss_weight[1] > eps:
                    loss_fl *= self.args.loss_weight[1]
                    loss += loss_fl
                losses[1] = loss_fl.item()

            if self.args.loss_weight[2] > -eps:
                # pixel loss
                loss_lpips = self.LPIPS.forward(rendered_pre_pix, self.rendered_ref_pix).sum()

                if self.args.loss_weight[2] > eps:
                    loss_lpips *= self.args.loss_weight[2]
                    loss += loss_lpips
                losses[2] = loss_lpips.item()

            if self.args.loss_weight[3] > -eps:
                # pixel loss
                loss_tex  = self.criterion(self.textures_init[:,0:3,:,:], textures_pre[:,0:3,:,:]) * 0.4
                loss_tex += self.criterion(self.textures_init[:,3:5,:,:], textures_pre[:,3:5,:,:]) * 0.1
                loss_tex += self.criterion(self.textures_init[:,5,:,:],   textures_pre[:,5,:,:])   * 0.1
                loss_tex += self.criterion(self.textures_init[:,6:9,:,:], textures_pre[:,6:9,:,:]) * 0.4

                lap = th.norm(self.Laplacian(self.textures_init - textures_pre).flatten())
                # print(lap.item())
                loss_tex += lap * 5e-5

                # print('loss_tex:', loss_tex)
                if self.args.loss_weight[3] > eps:
                    loss_tex *= self.args.loss_weight[3]
                    loss += loss_tex
                losses[3] = loss_tex.item()
        # print('loss_total:', loss)

        return loss, losses
