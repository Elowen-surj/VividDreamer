from audioop import mul
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler, \
                      EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, ControlNetModel, \
                      DDIMInverseScheduler, UNet2DConditionModel, LCMScheduler,PixArtAlphaPipeline, TCDScheduler, AutoPipelineForText2Image
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path
import os
import random

import torchvision.transforms as T
# suppress partial model loading warning
logging.set_verbosity_error()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator

from .sd_step import *
from tqdm.auto import tqdm
import time
import json
# from PIL import Image
# from ip_adapter.ip_adapter import IPAdapterPlus
# from ip_adapter.ip_adapter_faceid import IPAdapterFaceID

def rgb2sat(img, T=None):
    max_ = torch.max(img, dim=1, keepdim=True).values + 1e-5
    min_ = torch.min(img, dim=1, keepdim=True).values
    sat = (max_ - min_) / max_
    if T is not None:
        sat = (1 - T) * sat
    return sat

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    # np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, t_range=[0.02, 0.98], max_t_range=0.98, num_train_timesteps=None, 
                 ddim_inv=False, use_control_net=False, textual_inversion_path = None, 
                 LoRA_path = None, guidance_opt=None, iterations = 1000):
        super().__init__()

        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32

        print(f'[INFO] loading stable diffusion...')

        model_key = guidance_opt.model_key
        LCM_model_key = guidance_opt.LCM_model_key
        lcm_lora_id = guidance_opt.LCM_LoRA_key
        LCM_LoRA = guidance_opt.LCM_LoRA
        # assert model_key is not None

        is_safe_tensor = guidance_opt.is_safe_tensor
        is_LCM = guidance_opt.is_LCM
        base_model_key = "stabilityai/stable-diffusion-v1-5" if guidance_opt.base_model_key is None else guidance_opt.base_model_key # for finetuned model only

        if is_LCM:
            if is_safe_tensor:
                pipe = StableDiffusionPipeline.from_single_file(LCM_model_key, use_safetensors=True, torch_dtype=self.precision_t, load_safety_checker=False)
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            else:
                pipe = DiffusionPipeline.from_pretrained(LCM_model_key, torch_dtype=self.precision_t, variant="fp16")
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            # pipe.to(torch_dtype=self.precision_t)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)
        
        if LCM_LoRA:
            pipe.load_lora_weights(lcm_lora_id)
            pipe.fuse_lora()
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            
        elif guidance_opt.hyper_LoRA:
            pipe.load_lora_weights(lcm_lora_id)
            pipe.fuse_lora()
            pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

        self.iterations = iterations
        self.ism = not guidance_opt.sds

        pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config, torch_dtype=self.precision_t)
        # sss = DDIMInverseScheduler.from_pretrained(model_key if not is_safe_tensor else base_model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        # print('timesteps',self.scheduler.timesteps)
        # self.sche_func = ddim_step
        # breakpoint()

        if use_control_net:
            if guidance_opt.controlnet_type == 'depth':      
                controlnet_model_key = guidance_opt.controlnet_depth_model_key
                self.controlnet_depth = ControlNetModel.from_pretrained(controlnet_model_key,torch_dtype=self.precision_t).to(device)
            elif guidance_opt.controlnet_type == 'normal':
                controlnet_model_key = guidance_opt.controlnet_normal_model_key
                self.controlnet_normal = ControlNetModel.from_pretrained(controlnet_model_key,torch_dtype=self.precision_t).to(device)
            elif guidance_opt.controlnet_type == 'normal&depth':
                self.controlnet_depth = ControlNetModel.from_pretrained(guidance_opt.controlnet_depth_model_key,torch_dtype=self.precision_t).to(device)
                self.controlnet_normal = ControlNetModel.from_pretrained(guidance_opt.controlnet_normal_model_key,torch_dtype=self.precision_t).to(device)
                

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()

        self.sche_func = ddim_step

        pipe.enable_xformers_memory_efficient_attention()

        pipe = pipe.to(self.device)
        if textual_inversion_path is not None:
            pipe.load_textual_inversion(textual_inversion_path)
            print("load textual inversion in:.{}".format(textual_inversion_path))
        
        self.is_LCM = is_LCM
        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.num_train_timesteps = num_train_timesteps if num_train_timesteps is not None else self.pipe.scheduler.config.num_train_timesteps   
        

        self.timesteps = self.pipe.inverse_scheduler.timesteps
        self.timesteps = self.timesteps.to(device)
        self.error_data_sd = {}
        self.error_data_lcm = {}
            
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.warmup_step = int(self.num_train_timesteps*(max_t_range-t_range[1]))

        self.noise_temp = None
        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(guidance_opt.noise_seed)
        seed_everything(guidance_opt.noise_seed)

        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.rgb_latent_factors = torch.tensor([
                    # R       G       B
                    [ 0.298,  0.207,  0.208],
                    [ 0.187,  0.286,  0.173],
                    [-0.158,  0.189,  0.264],
                    [-0.184, -0.271, -0.473]
                ], device=self.device)

        

        print(f'[INFO] loaded stable diffusion!')
        # self.VSD = guidance_opt.VSD

    def augmentation(self, *tensors):
        augs = T.Compose([
                        T.RandomHorizontalFlip(p=0.5),
                    ])
        
        channels = [ten.shape[1] for ten in tensors]
        tensors_concat = torch.concat(tensors, dim=1)
        tensors_concat = augs(tensors_concat)

        results = []
        cur_c = 0
        for i in range(len(channels)):
            results.append(tensors_concat[:, cur_c:cur_c + channels[i], ...])
            cur_c += channels[i]
        return (ten for ten in results)

    def add_noise_with_cfg(self, latents, noise, 
                           ind_t, ind_prev_t, 
                           text_embeddings=None, cfg=1.0, 
                           delta_t=1, inv_steps=1,
                           is_noisy_latent=False,
                           eta=0.0):

        text_embeddings = text_embeddings.to(self.precision_t)
        if cfg <= 1.0:
            uncond_text_embedding = text_embeddings.reshape(2, -1, text_embeddings.shape[-2], text_embeddings.shape[-1])[1]

        unet = self.unet

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.pipe.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []

        for i in range(inv_steps):
            print('cur_t', self.timesteps[cur_ind_t])
            cur_noisy_lat_ = self.pipe.inverse_scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
            
            if cfg > 1.0:
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                unet_output = self.pipe.unet(latent_model_input, timestep_model_input, 
                                encoder_hidden_states=text_embeddings).sample
                
                uncond, cond = torch.chunk(unet_output, chunks=2)
                
                unet_output = cond + cfg * (uncond - cond) # reverse cfg to enhance the distillation
            else:
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat_.shape[0], 1).reshape(-1)

                unet_output = self.pipe.unet(cur_noisy_lat_, timestep_model_input, 
                                    encoder_hidden_states=uncond_text_embedding).sample

            pred_scores.append((cur_ind_t, unet_output))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t-cur_t

            cur_noisy_lat = self.sche_func(self.pipe.inverse_scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]


    @torch.no_grad()
    def get_text_embeds(self, prompt, resolution=(512, 512)):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def train_step_perpneg(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None, pred_normal=None,
                           grad_scale=1,use_control_net=False,
                           save_folder:Path=None, iteration=0, warm_up_rate = 0, weights = 0, 
                           resolution=(512, 512), guidance_opt=None,as_latent=False, embedding_inverse = None,
                           pose = None, step_ratio = 0):


        # flip aug
        # print(pred_rgb.shape,pred_depth.shape,pred_alpha.shape, pred_normal.shape)
        if guidance_opt.is_PixArt:
            resolution = (1024,1024)
        rgb_hw = pred_rgb.shape[2]
        start_time = time.time()
        # if len(pred_normal) != 0:
        #     if rgb_hw < resolution[0]:
        #         pred_rgb = F.interpolate(pred_rgb, (resolution[0], resolution[1]), mode="bilinear", align_corners=False)
        #         pred_depth = F.interpolate(pred_depth, (resolution[0], resolution[1]), mode="bilinear", align_corners=False)
        #         pred_alpha = F.interpolate(pred_alpha, (resolution[0], resolution[1]), mode="bilinear", align_corners=False)
        #         pred_normal = F.interpolate(pred_normal, (resolution[0], resolution[1]), mode="bilinear", align_corners=False)
        #     pred_rgb, pred_depth, pred_alpha, pred_normal = self.augmentation(pred_rgb, pred_depth, pred_alpha, pred_normal)
        # else:
        if rgb_hw < resolution[0]:
            pred_rgb = F.interpolate(pred_rgb, (resolution[0], resolution[1]), mode="bilinear", align_corners=False)
            pred_depth = F.interpolate(pred_depth, (resolution[0], resolution[1]), mode="bilinear", align_corners=False)
            pred_alpha = F.interpolate(pred_alpha, (resolution[0], resolution[1]), mode="bilinear", align_corners=False)
        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)
        
        # pose = pose.to(self.precision_t)

        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        if as_latent:      
            latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
        else:
            latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        # print('f', time.time() - start_time)
        weights = weights.reshape(-1)
        noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])

        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1]) # make it k+1, c * t, ...

        if iteration == guidance_opt.x0_iters:
            self.min_step = int(self.num_train_timesteps * guidance_opt.t_range_2[0])
            self.max_step = int(self.num_train_timesteps * guidance_opt.t_range_2[1])
            self.warmup_step = int(self.num_train_timesteps*(guidance_opt.max_t_range-guidance_opt.t_range_2[1]))

        if self.min_step == self.max_step + int(self.warmup_step*warm_up_rate):
            ind_t = torch.IntTensor([self.min_step])[0]
        if guidance_opt.timestep_ratio1 and iteration < guidance_opt.x0_iters:
        # elif guidance_opt.timestep_ratio:
            # if iteration < guidance_opt.x0_iters:
            step_ratio = iteration / guidance_opt.x0_iters
            ind_t = np.round(self.min_step + (1 - step_ratio) * (self.max_step - self.min_step)).clip(self.min_step, self.max_step)
            ind_t = torch.tensor(ind_t, dtype = torch.long)
        elif guidance_opt.timestep_ratio2 and iteration >= guidance_opt.x0_iters:
            step_ratio = (iteration - guidance_opt.x0_iters) / (self.iterations - guidance_opt.x0_iters)
            ind_t = np.round(self.min_step + (1 - step_ratio) * (self.max_step - self.min_step)).clip(self.min_step, self.max_step)
            ind_t = torch.tensor(ind_t, dtype = torch.long)
        else:
            ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        
        # if self.is_LCM:
        #     ind_t = ind_t_index * (self.num_train_timesteps // self.original_steps) - 1
        # else:
        # ind_t = ind_t_index

        t = self.timesteps[ind_t]
        add_noise_multi_steps = guidance_opt.add_noise_multi_steps
        with torch.no_grad():
            # step unroll via ddim inversion
            if not self.ism:
                if add_noise_multi_steps:
                    xs_delta_t = guidance_opt.xs_delta_t
                    if guidance_opt.xs_inv_steps_range is not None and iteration > guidance_opt.x0_iters:
                        step_ratio = (iteration - guidance_opt.x0_iters) / (self.iterations - guidance_opt.x0_iters)
                        xs_inv_steps = np.round(guidance_opt.xs_inv_steps_range[0] + (guidance_opt.xs_inv_steps_range[1] - guidance_opt.xs_inv_steps_range[0])*step_ratio).clip(guidance_opt.xs_inv_steps_range[0], guidance_opt.xs_inv_steps_range[1]).astype(int)
                    else:
                        xs_inv_steps = guidance_opt.xs_inv_steps

                    starting_t = max(t - xs_delta_t * xs_inv_steps, torch.ones_like(t) * 0)
                    # starting_ind = max(ind_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)
                    timesteps = list(range(starting_t, t, xs_delta_t))
                    # timesteps.remove(starting_t)
                    # timesteps.append(t)
                    
                    self.pipe.inverse_scheduler.set_timesteps(len(timesteps))
                    self.pipe.inverse_scheduler.timesteps = torch.tensor(timesteps)
                    if guidance_opt.denoise_guidance_scale <= 1.0:
                        uncond_inverse_text_embedding = inverse_text_embeddings.reshape(2, -1, inverse_text_embeddings.shape[-2], inverse_text_embeddings.shape[-1])[1]
                    # latents_noisy = latents
                    if guidance_opt.first_stage_DDIM:
                        latents_noisy = self.pipe.scheduler.add_noise(latents, noise, starting_t)
                        cur_t = starting_t
                        for i in range(xs_inv_steps):
                            
                            latents_noisy = self.pipe.inverse_scheduler.scale_model_input(latents_noisy, timestep=cur_t).to(self.precision_t)
                            with torch.no_grad():
                                # noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                                if guidance_opt.denoise_guidance_scale > 1.0:
                                    latent_model_input = torch.cat([latents_noisy, latents_noisy])
                                    timestep_model_input = cur_t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                                    unet_output = self.pipe.unet(latent_model_input, timestep_model_input, 
                                                    encoder_hidden_states=inverse_text_embeddings).sample
                                    
                                    uncond, cond = torch.chunk(unet_output, chunks=2)
                                    
                                    noise_pred = cond + guidance_opt.denoise_guidance_scale * (uncond - cond)
                                else:
                                    latent_model_input = latents_noisy
                                    noise_pred = self.pipe.unet(latent_model_input, cur_t, encoder_hidden_states=uncond_inverse_text_embedding).sample
                            
                            next_t = min(cur_t + xs_delta_t, t)
                            latents_noisy = self.sche_func(self.pipe.inverse_scheduler, noise_pred, cur_t, latents_noisy, cur_t - next_t, guidance_opt.xs_eta).prev_sample

                            cur_t = next_t

                            if cur_t == t:
                                break
                            
                        latents_noisy = latents_noisy * self.pipe.scheduler.init_noise_sigma
                    else:
                        if iteration > guidance_opt.x0_iters:
                            latents_noisy = self.pipe.scheduler.add_noise(latents, noise, starting_t)
                            cur_t = starting_t
                            for i in range(xs_inv_steps):

                                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                                # latent_model_input = torch.cat([img_latents] * 2)
                                # predict the noise residual
                                latents_noisy = self.pipe.inverse_scheduler.scale_model_input(latents_noisy, timestep=cur_t).to(self.precision_t)
                                with torch.no_grad():
                                    # noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                                    if guidance_opt.denoise_guidance_scale > 1.0:
                                        latent_model_input = torch.cat([latents_noisy, latents_noisy])
                                        timestep_model_input = cur_t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                                        unet_output = self.pipe.unet(latent_model_input, timestep_model_input, 
                                                        encoder_hidden_states=inverse_text_embeddings).sample
                                        
                                        uncond, cond = torch.chunk(unet_output, chunks=2)
                                        
                                        noise_pred = cond + guidance_opt.denoise_guidance_scale * (uncond - cond)
                                    else:
                                        latent_model_input = latents_noisy
                                        noise_pred = self.pipe.unet(latent_model_input, cur_t, encoder_hidden_states=uncond_inverse_text_embedding).sample
                                
                                # perform guidance
                                # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                                # compute the previous noisy sample x_t -> x_t-1
                                next_t = min(cur_t + xs_delta_t, t)
                                latents_noisy = self.sche_func(self.pipe.inverse_scheduler, noise_pred, cur_t, latents_noisy, cur_t - next_t, guidance_opt.xs_eta).prev_sample

                                cur_t = next_t

                                if cur_t == t:
                                    break
                                
                            latents_noisy = latents_noisy * self.pipe.scheduler.init_noise_sigma
                        else:
                            latents_noisy = self.pipe.scheduler.add_noise(latents, noise, t)
                    # _, latents_noisy, _ = self.add_noise_with_cfg(latents, noise, ind_t, starting_ind, inverse_text_embeddings, guidance_opt.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=guidance_opt.xs_eta)
                else:
                    latents_noisy = self.pipe.scheduler.add_noise(latents, noise, t)
                
                target = noise
        if self.is_LCM and guidance_opt.x0_loss:
            latents_noisy_copy = latents_noisy.clone().detach()
            with torch.no_grad():
                #####################################################################
                scheduler = self.pipe.scheduler if self.is_LCM else self.scheduler
                if guidance_opt.LCM_steps_num_range is not None and iteration > guidance_opt.x0_iters:
                    step_ratio = (iteration - guidance_opt.x0_iters) / (self.iterations - guidance_opt.x0_iters)
                    self.LCM_steps_num = np.round(guidance_opt.LCM_steps_num_range[0]+(guidance_opt.LCM_steps_num_range[1] - guidance_opt.LCM_steps_num_range[0])*step_ratio).clip(guidance_opt.LCM_steps_num_range[0], guidance_opt.LCM_steps_num_range[1]).astype(int)
                elif guidance_opt.LCM_steps_num_range1 is not None and iteration <= guidance_opt.x0_iters:
                    step_ratio = iteration / guidance_opt.x0_iters
                    self.LCM_steps_num = np.round(guidance_opt.LCM_steps_num_range1[0]+(guidance_opt.LCM_steps_num_range1[1] - guidance_opt.LCM_steps_num_range1[0])*step_ratio).clip(guidance_opt.LCM_steps_num_range1[0], guidance_opt.LCM_steps_num_range1[1]).astype(int)
                    # breakpoint()
                else:
                    self.LCM_steps_num = guidance_opt.LCM_steps_num
                self.LCM_step = guidance_opt.LCM_step
                self.fix_step = guidance_opt.fix_step
                # if ind_t_index < self.LCM_steps + 1:
                #     timesteps=((torch.linspace(ind_t_index * (self.num_train_timesteps // self.original_steps), 1, self.LCM_steps)).int() - 1).numpy()
                # else:
                # timesteps=((torch.linspace(ind_t_index, 1, self.LCM_steps)).int() * (self.num_train_timesteps // self.original_steps) - 1).numpy()
                # timesteps=(torch.linspace(t, 0, self.LCM_steps)).int().numpy()
                if self.fix_step:
                    timesteps = list(range((t.cpu().numpy() % self.LCM_step), t.cpu(), self.LCM_step))
                    timesteps.append(int(t.cpu().numpy()))
                    timesteps.reverse()
                    if 0 not in timesteps:
                        timesteps.append(0)
                else:
                    timesteps = np.linspace(t.cpu(), 0, self.LCM_steps_num, endpoint=guidance_opt.endpoint, dtype='int32')
                scheduler.set_timesteps(timesteps=timesteps)
                if iteration > guidance_opt.x0_iters and guidance_opt.guidance_scale2 != -1:
                    guidance_scale = guidance_opt.guidance_scale2
                else:
                    guidance_scale = guidance_opt.guidance_scale
                if guidance_opt.agg_first:
                    text_embeddings_copy = text_embeddings.clone().detach()
                    for timestep in scheduler.timesteps:
                        if timestep == scheduler.timesteps[0]:
                            latent_model_input = latents_noisy_copy[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
                        else:
                            latent_model_input = latents_noisy_copy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
                            if timestep == scheduler.timesteps[1]:
                                text_embeddings_copy = text_embeddings_copy.reshape(B, 1+K, text_embeddings.shape[-2], text_embeddings.shape[-1])
                                text_embeddings_copy = text_embeddings_copy[:,:2,...]
                                text_embeddings_copy = text_embeddings_copy.reshape(B*2,text_embeddings_copy.shape[-2], text_embeddings_copy.shape[-1])
                            
                        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
                        if use_control_net:
                            if guidance_opt.controlnet_type == 'depth':
                                pred_depth_input = pred_depth.to(self.precision_t)
                                if timestep == scheduler.timesteps[0]:
                                    pred_depth_input = pred_depth_input[None, :, ...].repeat(1 + K, 1, 3, 1, 1).reshape(-1, 3, resolution[0], resolution[1])
                                else:
                                    pred_depth_input = pred_depth_input[None, :, ...].repeat(2, 1, 3, 1, 1).reshape(-1, 3, resolution[0], resolution[1])
                                # print('pred_depth_input',pred_depth_input.shape)
                                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                                    latent_model_input.to(self.precision_t),
                                    timestep,
                                    encoder_hidden_states=text_embeddings_copy.to(self.precision_t),
                                    controlnet_cond=pred_depth_input,
                                    return_dict=False,
                                )
                            if guidance_opt.controlnet_type == 'normal':
                                pred_normal_input = pred_normal.to(self.precision_t)
                                if timestep == scheduler.timesteps[0]:
                                    pred_normal_input = pred_normal_input[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 3, resolution[0], resolution[1])
                                else:
                                    pred_normal_input = pred_normal_input[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 3, resolution[0], resolution[1])
                                # print('pred_depth_input',pred_depth_input.shape)
                                down_block_res_samples, mid_block_res_sample = self.controlnet_normal(
                                    latent_model_input.to(self.precision_t),
                                    timestep,
                                    encoder_hidden_states=text_embeddings_copy.to(self.precision_t),
                                    controlnet_cond=pred_normal_input,
                                    return_dict=False,
                                )
                                
                            unet_output = self.unet(
                                latent_model_input.to(self.precision_t), 
                                timestep, 
                                encoder_hidden_states=text_embeddings_copy.to(self.precision_t),
                                down_block_additional_residuals=down_block_res_samples, 
                                mid_block_additional_residual=mid_block_res_sample
                            ).sample
                            
                        else:
                            # breakpoint()
                            # print('latent_model_input',latent_model_input.shape)
                            unet_output = self.unet(
                                latent_model_input.to(self.precision_t), 
                                timestep, 
                                encoder_hidden_states=text_embeddings_copy.to(self.precision_t)
                            ).sample
                        if timestep == scheduler.timesteps[0]:
                            unet_output = unet_output.reshape(1 + K, -1, 4, resolution[0] // 8, resolution[1] // 8, )
                            noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
                            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
                            delta_DSD = weighted_perpendicular_aggregator(
                            delta_noise_preds,
                            weights,
                            B)
                            pred_noise = noise_pred_uncond + guidance_scale * delta_DSD
                            latent_model_input = latent_model_input[:1]
                        else:
                            noise_pred_uncond, noise_pred_text = unet_output.chunk(2)
                            noise_pred_uncond = noise_pred_uncond.reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
                            noise_pred_text = noise_pred_text.reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
                            delta_noise_preds = noise_pred_text - noise_pred_uncond
                            pred_noise = noise_pred_uncond + guidance_scale * delta_noise_preds
                            
                        latents_noisy_copy = scheduler.step(pred_noise, timestep, latents_noisy_copy).prev_sample
                        # print('f',time.time() - start_time)
                        
                else:
                    for timestep in scheduler.timesteps:
                        latent_model_input = latents_noisy_copy[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
                        
                        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
                        # print('text_embedding',text_embeddings.shape)
                        if use_control_net:
                            if guidance_opt.controlnet_type == 'depth':
                                pred_depth_input = pred_depth.to(self.precision_t)
                                pred_depth_input = pred_depth_input[None, :, ...].repeat(1 + K, 1, 3, 1, 1).reshape(-1, 3, resolution[0], resolution[1])
                                # print('pred_depth_input',pred_depth_input.shape)
                                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                                    latent_model_input.to(self.precision_t),
                                    timestep,
                                    encoder_hidden_states=text_embeddings.to(self.precision_t),
                                    controlnet_cond=pred_depth_input,
                                    return_dict=False,
                                )
                            if guidance_opt.controlnet_type == 'normal':
                                pred_normal_input = pred_normal.to(self.precision_t)
                                pred_normal_input = pred_normal_input[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 3, resolution[0], resolution[1])
                                # print('pred_depth_input',pred_depth_input.shape)
                                down_block_res_samples, mid_block_res_sample = self.controlnet_normal(
                                    latent_model_input.to(self.precision_t),
                                    timestep,
                                    encoder_hidden_states=text_embeddings.to(self.precision_t),
                                    controlnet_cond=pred_normal_input,
                                    return_dict=False,
                                )
                                
                            unet_output = self.unet(
                                latent_model_input.to(self.precision_t), 
                                timestep, 
                                encoder_hidden_states=text_embeddings.to(self.precision_t),
                                down_block_additional_residuals=down_block_res_samples, 
                                mid_block_additional_residual=mid_block_res_sample
                            ).sample
                            
                        else:
                            unet_output = self.unet(
                                latent_model_input.to(self.precision_t), 
                                timestep, 
                                encoder_hidden_states=text_embeddings.to(self.precision_t)
                            ).sample
                        unet_output = unet_output.reshape(1 + K, -1, 4, resolution[0] // 8, resolution[1] // 8, )
                        noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
                        delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
                        delta_DSD = weighted_perpendicular_aggregator(
                        delta_noise_preds,
                        weights,
                        B)
                        pred_noise = noise_pred_uncond + guidance_scale * delta_DSD
                        latents_noisy_copy = scheduler.step(pred_noise, timestep, latents_noisy_copy).prev_sample
                
                # assert False
                # pred_x0_t = self.decode_latents(latent_model_input.type(self.precision_t))

                #######################################################################################################
                # w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)
                w = lambda alphas: (1 - alphas)
                y = lambda alphas: (((1 - alphas) / alphas) ** 0.5)
            if guidance_opt.rgb_loss:
                grad = (pred_rgb - pred_x0_t)
                grad = torch.nan_to_num(grad_scale * grad)
                loss = SpecifyGradient.apply(pred_rgb, grad)
            else:
                grad = (latents - latents_noisy_copy)
                grad = torch.nan_to_num(grad_scale * grad)
                loss = SpecifyGradient.apply(latents, grad)
        
            pred_x0_t = self.decode_latents(latents_noisy_copy.type(self.precision_t))  

        if iteration % guidance_opt.vis_interval == 0:

            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            with torch.no_grad():


                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
                latents_sp_rgb = F.interpolate(lat2rgb(latents_noisy), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)


                if self.is_LCM and guidance_opt.x0_loss:
                    save_path_iter = os.path.join(save_folder,"iter_{}_step_{}_inv_steps_{}_LCM_steps_{}.jpg".format(iteration,t.item(),xs_inv_steps,self.LCM_steps_num))
                    viz_images = torch.cat([pred_rgb, 
                                            pred_depth.repeat(1, 3, 1, 1), 
                                            pred_alpha.repeat(1, 3, 1, 1),
                                            pred_x0_t,
                                            latents_rgb, latents_sp_rgb, 
                                            rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                            norm_grad],dim=0)
                    # if len(pred_normal) != 0:
                    #     viz_images = torch.cat([pred_rgb, 
                    #                         pred_depth.repeat(1, 3, 1, 1), 
                    #                         pred_alpha.repeat(1, 3, 1, 1), 
                    #                         pred_x0_t,
                    #                         pred_normal,
                    #                         rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                    #                         latents_rgb, latents_sp_rgb, 
                    #                         norm_grad],dim=0)
                save_image(viz_images, save_path_iter)


        return loss

    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor

        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs.to(target_dtype)

    def encode_imgs(self, imgs):
        target_dtype = imgs.dtype
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        kl_divergence = posterior.kl()

        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(target_dtype), kl_divergence