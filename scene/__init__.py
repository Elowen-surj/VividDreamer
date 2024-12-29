#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks,GenerateRandomCameras,GeneratePurnCameras,GenerateCircleCameras,GenerateCertainCameras
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, GenerateCamParams, PipelineParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, cameraList_from_RcamInfos, orbit_camera, safe_normalize, perspective
import torch
from scene.cameras import MiniCam2
import numpy as np
import torch.nn.functional as F
# from grid_put import mipmap_linear_grid_put_2d
from utils.graphics_utils import focal2fov, fov2focal


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, pose_args : GenerateCamParams, pipe: PipelineParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args._model_path
        self.pretrained_model_path = args.pretrained_model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.resolution_scales = resolution_scales
        self.pose_args = pose_args
        self.args = args
        self.pipe = pipe
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.test_cameras = {}
        scene_info = sceneLoadTypeCallbacks["RandomCam"](self.model_path , pose_args)

        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = pose_args.default_radius #    scene_info.nerf_normalization["radius"]
        for resolution_scale in resolution_scales:
            self.test_cameras[resolution_scale] = cameraList_from_RcamInfos(scene_info.test_cameras, resolution_scale, self.pose_args)
        print(pose_args.init_ply_path)
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        elif self.pretrained_model_path is not None:
            self.gaussians.load_ply(self.pretrained_model_path)
        elif pose_args.init_ply_path is not None:
            self.gaussians.load_ply_2(pose_args.init_ply_path)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, bg_color : torch.Tensor, cam = {}, mode = '', density_thresh = 1, texture_size = 1024):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        mesh_path = os.path.join(self.model_path, "mesh/iteration_{}".format(iteration))
        if mode == 'geo':
            os.makedirs(mesh_path, exist_ok=True)
            path = os.path.join(mesh_path, '_mesh.ply')
            mesh = self.gaussians.extract_mesh(path, density_thresh)
            mesh.write_ply(path)
        elif mode == 'geo+tex':
            os.makedirs(mesh_path, exist_ok=True)
            path = path = os.path.join(mesh_path, '_mesh.obj')
            mesh = self.gaussians.extract_mesh(path, density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.args.data_device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.args.data_device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            # if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
            glctx = dr.RasterizeGLContext()
            # else:
            #     glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.pose_args.default_radius)

                fov = self.pose_args.default_fovy
                fovy = focal2fov(fov2focal(fov, self.pose_args.image_h), self.pose_args.image_w)
                fovx = fov
                far = 100.0
                near = 0.01
                cur_cam = MiniCam2(
                    pose,
                    render_resolution,
                    render_resolution,
                    fovy,
                    fovx,
                    near,
                    far,
                )
                from gaussian_renderer import render
                cur_out = render(cur_cam, self.gaussians, self.pipe, bg_color, 
                                sh_deg_aug_ratio = self.args.sh_deg_aug_ratio, 
                                bg_aug_ratio = self.args.bg_aug_ratio, 
                                shs_aug_ratio = self.args.shs_aug_ratio, 
                                scale_aug_ratio = self.args.scale_aug_ratio, return_normal = False)

                rgbs = cur_out["render"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.args.data_device)
                proj = torch.from_numpy(perspective(fovy, W = self.pose_args.image_w, H = self.pose_args.image_h, far = far, near = near).astype(np.float32)).to(self.args.data_device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.args.data_device)
            mesh.write(path)
        else:
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        

    def getRandTrainCameras(self, scale=1.0):
        rand_train_cameras = GenerateRandomCameras(self.pose_args, self.args.batch, SSAA=True)
        train_cameras = {}
        for resolution_scale in self.resolution_scales:
            train_cameras[resolution_scale] = cameraList_from_RcamInfos(rand_train_cameras, resolution_scale, self.pose_args, SSAA=True)
        return train_cameras[scale]
    
    def getCertainTrainCameras(self, azimuth = 0, polar = 90, scale=1.0):
        rand_train_cameras = GenerateCertainCameras(self.pose_args, self.args.batch, SSAA=True, azimuth = azimuth, polar = polar)
        train_cameras = {}
        for resolution_scale in self.resolution_scales:
            train_cameras[resolution_scale] = cameraList_from_RcamInfos(rand_train_cameras, resolution_scale, self.pose_args, SSAA=True)
        return train_cameras[scale]


    def getPurnTrainCameras(self, scale=1.0):
        rand_train_cameras = GeneratePurnCameras(self.pose_args)
        train_cameras = {}
        for resolution_scale in self.resolution_scales:
            train_cameras[resolution_scale] = cameraList_from_RcamInfos(rand_train_cameras, resolution_scale, self.pose_args)        
        return train_cameras[scale]


    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getCircleVideoCameras(self, scale=1.0,batch_size=120, render45 = True):
        video_circle_cameras = GenerateCircleCameras(self.pose_args,batch_size,render45)
        video_cameras = {}
        for resolution_scale in self.resolution_scales:
            video_cameras[resolution_scale] = cameraList_from_RcamInfos(video_circle_cameras, resolution_scale, self.pose_args)        
        return video_cameras[scale]