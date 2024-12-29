import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
import numpy as np
import os
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh
import open3d as o3d
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH,SH2RGB

def init_from_pointe(prompt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('creating base model...')
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))
    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
    )
    # Produce a sample from the model.
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]), samples=samples)):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]
    xyz = pc.coords
    rgb = np.zeros_like(xyz)
    rgb[:,0],rgb[:,1],rgb[:,2] = pc.channels['R'],pc.channels['G'],pc.channels['B']
    return xyz,rgb

def init_from_finetuned_pointe(prompt):
    base_path = 'point_e_model_cache'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('creating base model...')
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
    
    print('loading finetuned checkpoint...')
    base_model.load_state_dict(torch.load(os.path.join(base_path, 'pointE_finetuned_with_330kdata.pth'), map_location=device)['model_state_dict'])

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
    )
    # Produce a sample from the model.
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]), samples=samples)):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]
    xyz = pc.coords
    rgb = np.zeros_like(xyz)
    rgb[:,0],rgb[:,1],rgb[:,2] = pc.channels['R'],pc.channels['G'],pc.channels['B']
    return xyz,rgb

def init_from_finetuned_shape(prompt):
    base_path = 'shap_e_model_cache'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    model.load_state_dict(torch.load(os.path.join(base_path, 'shapE_finetuned_with_330kdata.pth'), map_location=device)['model_state_dict'])
    diffusion = diffusion_from_config_shape(load_config('diffusion'))

    batch_size = 1
    guidance_scale = 15.0
    print('prompt1',prompt)
    prompt = str(prompt)
    print('prompt2',prompt)

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    pc = decode_latent_mesh(xm, latents[0]).tri_mesh()
    
    skip = 1
    coords = pc.verts
    rgb = np.concatenate([pc.vertex_channels['R'][:,None],pc.vertex_channels['G'][:,None],pc.vertex_channels['B'][:,None]],axis=1) 

    coords = coords[::skip]
    rgb = rgb[::skip]
    
    
    return coords,rgb

def upsample_pointe(init_points, num_pts, prompt):
    print('init_points',init_points.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('creating base model...')
    # base_name = 'base40M-textvec'
    # base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    # base_model.eval()
    # base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    init_points = torch.FloatTensor(init_points).to(device)
    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    # print('downloading base checkpoint...')
    # base_model.load_state_dict(load_checkpoint(base_name, device))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    sampler = PointCloudSampler(
        device=device,
        models=[upsampler_model],
        diffusions=[upsampler_diffusion],
        num_points=[num_pts - init_points.shape[2]],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[0.0],
        use_karras = [True],
        karras_steps=[64], # mytest
        model_kwargs_key_filter=(''), # Do not condition the upsampler at all
        sigma_min=[1e-3],
        sigma_max=[160],
        s_churn=[0],
    )
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]), samples=init_points)):
        # print(count)
        # if samples is not None:
        #     print(samples.shape)
        samples = x
        # if count == 64:
        #     p = samples.detach().cpu().numpy()
        #     np.save('p.npy', p)
            
        # count += 1
    pc = sampler.output_to_point_clouds(samples)[0]
    xyz = pc.coords
    rgb = np.zeros_like(xyz)
    rgb[:,0],rgb[:,1],rgb[:,2] = pc.channels['R'],pc.channels['G'],pc.channels['B']
    return xyz,rgb

def init_from_obj(obj_path):
    num_pts = 100000
    mesh = o3d.io.read_triangle_mesh(obj_path)
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_pts)
    coords = np.array(point_cloud.points)
    rgb = np.random.random((num_pts, 3)) / 255.0
    adjusment = np.zeros_like(coords)
    adjusment[:,0] = coords[:,0]
    adjusment[:,1] = coords[:,2]
    adjusment[:,2] = coords[:,1]
    current_center = np.mean(adjusment, axis=0)
    center_offset = -current_center
    adjusment += center_offset
    return adjusment, rgb


def init_from_ply(ply_path):
    plydata = PlyData.read(ply_path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["z"]),
                        np.asarray(plydata.elements[0]["y"])),  axis=1)
    xyz[:,0] = -xyz[:,0]
    pts = xyz
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    rgb = SH2RGB(features_dc)
    # breakpoint()
    rgb = np.squeeze(rgb, axis = 2)
    return pts, rgb
    
    