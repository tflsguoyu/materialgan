import numpy as np
import torch
import sys
import os

import copy

import global_var
from util import *
from render import *

sys.path.insert(1, 'higan/models/')
from stylegan2_generator import StyleGAN2Generator


def save_image(output, save_dir, save_name, make_dir=False, is_naive=False):

    if make_dir:
        save_dir = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    all_images_np = output['image']  # shape [num_interps, H, W, C]

    renders = []
    material_maps = []
    for i in range(len(all_images_np)):
        maps = all_images_np[i]  # shape [B, C, H, W]
        render = render_map(maps)

        material_maps.append(maps)
        renders.append(render)
    images_np = np.hstack(renders)
    images_pil = gyArray2PIL(images_np)

    if is_naive:
        save_path = os.path.join(save_dir, save_name + "_renders_NAIVE.png")
    else:
        save_path = os.path.join(save_dir, save_name+"_renders_GAN.png")
    images_pil.save(save_path)

    all_material_maps = []
    for material_map in material_maps:
        map_t = torch.from_numpy(material_map)
        map_t = torch.unsqueeze(map_t, 0)
        albedo, normal, roughness, specular = tex2map(map_t)

        albedo = gyTensor2Array(albedo[0, :].permute(1, 2, 0))
        normal = gyTensor2Array((normal[0, :].permute(1, 2, 0) + 1) / 2)
        roughness = gyTensor2Array(roughness[0, :].permute(1, 2, 0))
        specular = gyTensor2Array(specular[0, :].permute(1, 2, 0))

        material_maps = np.vstack([albedo, normal, roughness, specular])
        all_material_maps.append(material_maps)

    materials_np = np.hstack(all_material_maps)
    materials_pil = gyArray2PIL(materials_np)

    if is_naive:
        save_path = os.path.join(save_dir, save_name + "_materials_NAIVE.png")
    else:
        save_path = os.path.join(save_dir, save_name+"_materials_GAN.png")
    materials_pil.save(save_path)



def generateLightCameraPosition(p, angle, colocated=True, addNoise=True):
    theta = (np.pi/180 * np.array([0] + [angle]*8)).astype(np.float32)
    phi   = (np.pi/4   * np.array([0,1,5,3,7,2,6,4,8])).astype(np.float32)

    light_pos = np.stack((p * np.sin(theta) * np.cos(phi),
                          p * np.sin(theta) * np.sin(phi),
                          p * np.cos(theta))).transpose()
    if addNoise:
        light_pos[:,0:2] += np.random.randn(9,2).astype(np.float32)

    if colocated:
        camera_pos = light_pos.copy()
    else:
        camera_pos = np.array([[0,0,p]]).astype(np.float32).repeat(9,axis=0)
        if addNoise:
            camera_pos[:,0:2] += np.random.randn(9,2).astype(np.float32)

    return light_pos[0], camera_pos[0]


def render_map(img_np):
    light_position_np, camera_position_np = generateLightCameraPosition(20, 20, True, False)
    light_intensity_np = np.array([1500.0, 1500.0, 1500.0])

    with torch.no_grad():
        light_intensity_t = torch.from_numpy(light_intensity_np).cuda()

        img_t = torch.from_numpy(img_np)
        img_t = torch.unsqueeze(img_t, 0).cuda()

        microfacetObj = Microfacet(res=256, size=20)
        render = microfacetObj.eval(img_t, light_position_np, camera_position_np, light_intensity_t)
        render = render[0].detach().cpu().numpy()
        render = np.transpose(render, (1, 2, 0))
        render = gyApplyGamma(render, 1 / 2.2)

        return np.clip(render, 0, 1)


def lerp_np(v0, v1, num_interps):
    if v0.shape != v1.shape:
        raise ValueError('A and B must have the same shape to interpolate.')
    alphas = np.linspace(0, 1, num_interps)
    return np.array([(1-a)*v0 + a*v1 for a in alphas])


def lerp(v0, v1, t):
    return (1-t)*v0 + t*v1


def lerp_noises(noises1, noises2, t):
    noises = []
    for noise1, noise2 in zip(noises1, noises2):
        noise_lerp = torch.lerp(noise1, noise2, t)
        noises.append(noise_lerp)
    return noises


def slerp(v0, v1, num_interps):
    """Spherical linear interpolation."""
    # >>> slerp([1,0,0,0], [0,0,0,1], np.arange(0, 1, 0.001))
    t_array = np.arange(0, 1, 1/num_interps)
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0 * v1)

    if dot < 0.0:
        v1 = -v1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = v0[np.newaxis, :] + t_array[:, np.newaxis] * (v1 - v0)[np.newaxis, :]
        return (result.T / np.linalg.norm(result, axis=1)).T

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t_array
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:, np.newaxis] * v0[np.newaxis, :]) + (s1[:, np.newaxis] * v1[np.newaxis, :])


def interpolate_random_textures(num_maps,
                                save_dir,
                                num_interps=10,
                                resolution=256,
                                noise_init="random",
                                search_type="brute",
                                interp_type="lerp",
                                latent_space_type='z'):
    global_var.init_global_noise(resolution, noise_init)
    genObj = StyleGAN2Generator('svbrdf')

    if search_type == "brute":

        z = genObj.sample(num_maps)
        w = genObj.synthesize(z)['z']

        nearest_neighbor_idx = [-1] * (len(w) - 1)
        for i in range(0, len(w)-1, 1):
            dist = sys.float_info.max
            for j in range(i + 1, len(w), 1):
                new_dist = np.linalg.norm(w[i] - w[j])
                if new_dist < dist:
                    nearest_neighbor_idx[i] = j

        # (n*(n-1)) / 2 pairs
        for i in range(0, len(nearest_neighbor_idx), 1):
            j = nearest_neighbor_idx[i]
            if interp_type == "lerp":
                interp = lerp_np(w[i], w[j], num_interps=num_interps)
            elif interp_type == "slerp":
                interp = slerp(w[i], w[j], num_interps=num_interps)
            outputs = genObj.synthesize(interp, latent_space_type='z')

            # save images
            save_name = "{}_{}".format(i, j)
            save_image(outputs, save_dir, save_name)

    elif search_type == "random":

        z0 = genObj.sample(num_maps)
        w0 = genObj.synthesize(z0)['w']

        z1 = genObj.sample(num_maps)
        w1 = genObj.synthesize(z1)['w']

        sorted_dist = []
        for i in range(0, len(w0), 1):
            w1_sorted = sorted(w1, key=lambda e: np.linalg.norm(w0[i] - e))
            sorted_dist.append(w1_sorted)

        for i in range(len(w0)):
            w1 = sorted_dist[i]
            j = np.random.randint(0, len(w1) // 2) # random from nearest half
            interp = lerp_np(w0[i], w1[j], num_interps=num_interps)
            outputs = genObj.synthesize(interp, latent_space_type='w')

            # save images
            save_name = "{}_{}".format(i, j)
            save_image(outputs, save_dir, save_name)


def interpolate_projected_textures(latent_paths,
                                   noises_paths,
                                   image_paths,
                                   save_dir,
                                   num_interps=10,
                                   resolution=256,
                                   search_type="brute",
                                   num_maps=10):

    global_var.init_global_noise(resolution, "random")
    genObj = StyleGAN2Generator('svbrdf')

    all_latents, all_noises = [], []

    for latent_path in latent_paths:
        latent = torch.load(latent_path).detach().cpu().numpy()
        all_latents.append(latent)

    for noises_path in noises_paths:
        noises_list = torch.load(noises_path)
        noise_vars = []
        for noise in noises_list:
            noise = noise.cuda()
            noise_vars.append(noise)
        all_noises.append(noise_vars)

    all_images = []
    for image_path in image_paths:
        map, _ = png2tex(image_path)
        map_np = map.detach().cpu().numpy()
        all_images.append(map_np[0])

    if search_type == "brute":

        ts = np.linspace(0., 1., num_interps)

        for i in range(0, len(all_latents) - 1, 1):

            name1 = os.path.basename(os.path.dirname(latent_paths[i]))

            for j in range(i + 1, len(all_latents), 1):

                name2 = os.path.basename(os.path.dirname(latent_paths[j]))

                lerp_outputs = []
                for t in ts:
                    noises = lerp_noises(all_noises[i], all_noises[j], t)
                    lerp_latent = lerp(all_latents[i], all_latents[j], t)

                    global_var.noises = noises
                    outputs = genObj.synthesize(lerp_latent, latent_space_type="wp")
                    lerp_outputs.append(outputs['image'])
                lerp_outputs = np.concatenate(lerp_outputs, 0)

                lerp_maps = lerp_np(all_images[i], all_images[j], num_interps=num_interps)
                # save images
                save_name = "{}_VERSUS_{}".format(name1, name2)
                save_image({'image': lerp_outputs}, save_dir, save_name, make_dir=True)
                save_image({'image': lerp_maps}, save_dir, save_name, make_dir=True, is_naive=True)

    elif search_type == "random":

        rand_z = genObj.sample(num_maps)

        def rand_noise():
            global_var.init_global_noise(resolution, "random")
            return copy.deepcopy(global_var.noises)
        rand_noises = [rand_noise() for _ in range(len(rand_z))]

        rand_latents = genObj.synthesize(rand_z, latent_space_type="z")['wp']
        ts = np.linspace(0., 1., num_interps)

        for i in range(0, len(all_latents) - 1, 1):

            name1 = os.path.basename(os.path.dirname(latent_paths[i]))

            for j in range(0, len(rand_z), 1):
                lerp_outputs = []
                for t in ts:
                    noises = lerp_noises(all_noises[i], rand_noises[j], t)

                    global_var.noises = noises

                    lerp_latent = lerp(all_latents[i][0], rand_latents[j], t)

                    lerp_latent = np.expand_dims(lerp_latent, 0)
                    outputs = genObj.synthesize(lerp_latent, latent_space_type="wp")
                    lerp_outputs.append(outputs['image'])

                lerp_outputs = np.concatenate(lerp_outputs, 0)

                # save images
                save_name = "{}_{}".format(name1, j)
                save_image({'image': lerp_outputs}, save_dir, save_name)

