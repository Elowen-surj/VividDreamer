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
import sys
import torch
import random
import torch.nn.functional as F
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from utils.pointe_utils import init_from_pointe, init_from_finetuned_pointe, init_from_finetuned_shape, upsample_pointe, init_from_obj, init_from_ply
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.general_utils import inverse_sigmoid_np
from scene.gaussian_model import BasicPointCloud
import open3d as o3d


class RandCameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    width: int
    height: int 
    delta_polar : np.array
    delta_azimuth : np.array
    delta_radius : np.array
    pose : np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


class RSceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    test_cameras: list
    ply_path: str
    
class RSceneInfo_1(NamedTuple):
    test_cameras: list
    ply_path: str

# def getNerfppNorm(cam_info):
#     def get_center_and_diag(cam_centers):
#         cam_centers = np.hstack(cam_centers)
#         avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
#         center = avg_cam_center
#         dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
#         diagonal = np.max(dist)
#         return center.flatten(), diagonal

#     cam_centers = []

#     for cam in cam_info:
#         W2C = getWorld2View2(cam.R, cam.T)
#         C2W = np.linalg.inv(W2C)
#         cam_centers.append(C2W[:3, 3:4])

#     center, diagonal = get_center_and_diag(cam_centers)
#     radius = diagonal * 1.1

#     translate = -center

#     return {"translate": translate, "radius": radius}

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

def fetchPly_stage2(path, upsample = False, num_pts=5000, prompt='', color = True):
    print(path)
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    # colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    print('positions',positions.shape)
    features_dc = np.zeros((positions.shape[0], 3))
    features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
    colors = SH2RGB(features_dc)
    print('colors',colors.shape)
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    print('normals:',normals.shape, np.unique(normals))
    # normals1 = np.random.rand(num_pts, 3)
    # normals = normals + normals1
    if upsample:
        # sample = np.zeros((1, 6, positions.shape[0]))
        # sample[:,:3] = np.expand_dims(positions.transpose(1,0), axis=0)
        # sample[:,:3] = np.expand_dims(colors.transpose(1,0), axis=0)
        # positions, colors = upsample_pointe(sample, num_pts, prompt)
        # normals = np.zeros((num_pts, 3))
        point_num = positions.shape[0]
        num_pts = int(num_pts // point_num)
        print('num_pts',num_pts)
        thetas = np.random.rand(num_pts)*np.pi
        phis = np.random.rand(num_pts)*2*np.pi        
        radius = np.random.rand(num_pts)*0.05
        # We create random points inside the bounds of sphere
        xyz_ball = np.stack([
            radius * np.sin(thetas) * np.sin(phis),
            radius * np.sin(thetas) * np.cos(phis),
            radius * np.cos(thetas),
        ], axis=-1) # [B, 3]expend_dims
        rgb_ball = np.random.random((point_num, num_pts, 3))*0.0001
        colors = (np.expand_dims(colors,axis=1)+rgb_ball).reshape(-1,3)
        positions = (np.expand_dims(positions,axis=1)+np.expand_dims(xyz_ball,axis=0)).reshape(-1,3)
        positions = positions * 1.
        normals = np.zeros((positions.shape[0], 3))
    if not color:    
        shs = np.random.random((positions.shape[0], 3)) / 255.0
        pcd = BasicPointCloud(points=positions, colors=SH2RGB(shs), normals=normals)
    else:
        pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
    return pcd

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

#only test_camera
def readCircleCamInfo(path,opt):
    print("Reading Test Transforms")
    test_cam_infos = GenerateCircleCameras(opt,render45 = opt.render_45)
    upsample = False
    # if opt.init_ply_path != None:
    #     ply_path = opt.init_ply_path
    #     # upsample = True
    #     pcd = fetchPly_stage2(ply_path, upsample = upsample, num_pts = opt.init_num_pts, prompt = opt.prompt, color = opt.use_stage1_color)
    # else:
    if opt.init_ply_path == None:
        ply_path = os.path.join(path, "init_points3d.ply")
        if os.path.exists(ply_path):
            pcd = fetchPly(ply_path)
        print('ply_path',ply_path)
    else:
        ply_path = opt.init_ply_path
        scene_info = RSceneInfo_1(test_cameras=test_cam_infos,
                           ply_path=ply_path)
        return scene_info
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = opt.init_num_pts  
        if opt.init_shape == 'sphere':
            thetas = np.random.rand(num_pts)*np.pi
            phis = np.random.rand(num_pts)*2*np.pi        
            radius = np.random.rand(num_pts)*0.5
            # We create random points inside the bounds of sphere
            xyz = np.stack([
                radius * np.sin(thetas) * np.sin(phis),
                radius * np.sin(thetas) * np.cos(phis),
                radius * np.cos(thetas),
            ], axis=-1) # [B, 3]
        elif opt.init_shape == 'box':
            xyz = np.random.random((num_pts, 3)) * 1.0 - 0.5
        elif opt.init_shape == 'rectangle_x':
            xyz = np.random.random((num_pts, 3))
            xyz[:, 0] = xyz[:, 0] * 0.6 - 0.3
            xyz[:, 1] = xyz[:, 1] * 1.2 - 0.6
            xyz[:, 2] = xyz[:, 2] * 0.5 - 0.25
        elif opt.init_shape == 'rectangle_y':
            xyz = np.random.random((num_pts, 3))
            xyz[:, 0] = xyz[:, 0] * 1.2 - 0.6
            xyz[:, 1] = xyz[:, 1] * 0.6 - 0.3
            xyz[:, 2] = xyz[:, 2] * 0.6 - 0.3
        elif opt.init_shape == 'rectangle_z':
            xyz = np.random.random((num_pts, 3))
            xyz[:, 0] = xyz[:, 0] * 1.0 - 0.5
            xyz[:, 1] = xyz[:, 1] * 0.1 - 0.05
            xyz[:, 2] = xyz[:, 2] * 1.0 - 0.5
        elif opt.init_shape == 'pointe':
            num_pts = int(num_pts/5000)
            xyz,rgb = init_from_pointe(opt.init_prompt)
            xyz[:,1] = - xyz[:,1]
            xyz[:,2] = xyz[:,2] + 0.15
            thetas = np.random.rand(num_pts)*np.pi
            phis = np.random.rand(num_pts)*2*np.pi        
            radius = np.random.rand(num_pts)*0.05
            # We create random points inside the bounds of sphere
            xyz_ball = np.stack([
                radius * np.sin(thetas) * np.sin(phis),
                radius * np.sin(thetas) * np.cos(phis),
                radius * np.cos(thetas),
            ], axis=-1) # [B, 3]expend_dims
            rgb_ball = np.random.random((4096, num_pts, 3))*0.0001
            rgb = (np.expand_dims(rgb,axis=1)+rgb_ball).reshape(-1,3)
            xyz = (np.expand_dims(xyz,axis=1)+np.expand_dims(xyz_ball,axis=0)).reshape(-1,3)
            xyz = xyz * 1.
            num_pts = xyz.shape[0]
        elif opt.init_shape == 'pointe_finetune':
            num_pts = int(num_pts/5000)
            xyz,rgb = init_from_finetuned_pointe(opt.init_prompt)
            xyz[:,1] = - xyz[:,1]
            xyz[:,2] = xyz[:,2] + 0.15
            if opt.upsample_init_pointc == 'knn':
                print('knn')
                coords = xyz
                pcd_by3d = o3d.geometry.PointCloud()
                pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
                
                bbox = pcd_by3d.get_axis_aligned_bounding_box()
                np.random.seed(0)
    
                points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_pts, 3))

                kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)

                points_inside = []
                color_inside= []
                for point in points:
                    _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
                    nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
                    if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
                        points_inside.append(point)
                        color_inside.append(rgb[idx[0]]+0.2*np.random.random(3))

                all_coords = np.array(points_inside)
                all_rgb = np.array(color_inside)
                if not len(all_coords) == 0:
                    all_coords = np.concatenate([all_coords,coords],axis=0)
                    all_rgb = np.concatenate([all_rgb,rgb],axis=0)
                else:
                    all_coords = coords
                    all_rgb = rgb
                num_pts = all_coords.shape[0]
                # print('all_coords',all_coords)
                xyz = all_coords * 0.6
                rgb = all_rgb
            else:
                thetas = np.random.rand(num_pts)*np.pi
                phis = np.random.rand(num_pts)*2*np.pi        
                radius = np.random.rand(num_pts)*0.05
                # We create random points inside the bounds of sphere
                xyz_ball = np.stack([
                    radius * np.sin(thetas) * np.sin(phis),
                    radius * np.sin(thetas) * np.cos(phis),
                    radius * np.cos(thetas),
                ], axis=-1) # [B, 3]expend_dims
                rgb_ball = np.random.random((4096, num_pts, 3))*0.0001
                rgb = (np.expand_dims(rgb,axis=1)+rgb_ball).reshape(-1,3)
                xyz = (np.expand_dims(xyz,axis=1)+np.expand_dims(xyz_ball,axis=0)).reshape(-1,3)
                xyz = xyz * 1.
                num_pts = xyz.shape[0]
            
        elif opt.init_shape == 'shape_finetune':
            num_pts = int(num_pts/5000)
            xyz,rgb= init_from_finetuned_shape(opt.init_prompt)
            print(xyz.shape)
            xyz[:,1] = - xyz[:,1]
            if opt.upsample_init_pointc == 'knn':
                print('knn')
                xyz = xyz * 1.2
                coords = xyz
                pcd_by3d = o3d.geometry.PointCloud()
                pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
                
                bbox = pcd_by3d.get_axis_aligned_bounding_box()
                np.random.seed(0)
    
                points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_pts, 3))

                kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)

                points_inside = []
                color_inside= []
                for point in points:
                    _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
                    nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
                    if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
                        points_inside.append(point)
                        color_inside.append(rgb[idx[0]]+0.2*np.random.random(3))

                all_coords = np.array(points_inside)
                all_rgb = np.array(color_inside)
                if not len(all_coords) == 0:
                    all_coords = np.concatenate([all_coords,coords],axis=0)
                    all_rgb = np.concatenate([all_rgb,rgb],axis=0)
                else:
                    all_coords = coords
                    all_rgb = rgb
                num_pts = all_coords.shape[0]
                # print('all_coords',all_coords)
                xyz = all_coords * 0.6
                rgb = all_rgb
            else:
                thetas = np.random.rand(num_pts)*np.pi
                phis = np.random.rand(num_pts)*2*np.pi        
                radius = np.random.rand(num_pts)*0.05
                # We create random points inside the bounds of sphere
                xyz_ball = np.stack([
                    radius * np.sin(thetas) * np.sin(phis),
                    radius * np.sin(thetas) * np.cos(phis),
                    radius * np.cos(thetas),
                ], axis=-1) # [B, 3]expend_dims
                rgb_ball = np.random.random((4096, num_pts, 3))*0.0001
                rgb = (np.expand_dims(rgb,axis=1)+rgb_ball).reshape(-1,3)
                xyz = (np.expand_dims(xyz,axis=1)+np.expand_dims(xyz_ball,axis=0)).reshape(-1,3)
                xyz = xyz * 1.
                num_pts = xyz.shape[0]
            
        elif opt.init_shape == 'scene':
            thetas = np.random.rand(num_pts)*np.pi
            phis = np.random.rand(num_pts)*2*np.pi        
            radius = np.random.rand(num_pts) + opt.radius_range[-1]*3
            # We create random points inside the bounds of sphere
            xyz = np.stack([
                radius * np.sin(thetas) * np.sin(phis),
                radius * np.sin(thetas) * np.cos(phis),
                radius * np.cos(thetas),
            ], axis=-1) # [B, 3]
        elif opt.init_shape == 'obj':
            num_pts = int(num_pts/50000)
            xyz,rgb= init_from_obj(opt.init_path)
            xyz = xyz * 1.3
            print(xyz.shape)
            # xyz[:,1] = - xyz[:,1]
            if opt.upsample_init_pointc == 'knn':
                print('knn')
                xyz = xyz * 1.2
                coords = xyz
                pcd_by3d = o3d.geometry.PointCloud()
                pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
                
                bbox = pcd_by3d.get_axis_aligned_bounding_box()
                np.random.seed(0)
    
                points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_pts, 3))

                kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)

                points_inside = []
                color_inside= []
                for point in points:
                    _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
                    nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
                    if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
                        points_inside.append(point)
                        color_inside.append(rgb[idx[0]]+0.2*np.random.random(3))

                all_coords = np.array(points_inside)
                all_rgb = np.array(color_inside)
                if not len(all_coords) == 0:
                    all_coords = np.concatenate([all_coords,coords],axis=0)
                    all_rgb = np.concatenate([all_rgb,rgb],axis=0)
                else:
                    all_coords = coords
                    all_rgb = rgb
                num_pts = all_coords.shape[0]
                # print('all_coords',all_coords)
                xyz = all_coords * 0.6
                rgb = all_rgb
        elif opt.init_shape == 'ply':
            num_pts = int(num_pts/50000)
            xyz,rgb= init_from_ply(opt.init_path)
            # xyz = xyz * 1.3
            print(xyz.shape)
            # xyz[:,1] = - xyz[:,1]
            if opt.upsample_init_pointc == 'knn':
                print('knn')
                xyz = xyz * 1.2
                coords = xyz
                pcd_by3d = o3d.geometry.PointCloud()
                pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
                
                bbox = pcd_by3d.get_axis_aligned_bounding_box()
                np.random.seed(0)
    
                points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_pts, 3))

                kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)

                points_inside = []
                color_inside= []
                for point in points:
                    _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
                    nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
                    if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
                        points_inside.append(point)
                        color_inside.append(rgb[idx[0]]+0.2*np.random.random(3))

                all_coords = np.array(points_inside)
                all_rgb = np.array(color_inside)
                if not len(all_coords) == 0:
                    all_coords = np.concatenate([all_coords,coords],axis=0)
                    all_rgb = np.concatenate([all_rgb,rgb],axis=0)
                else:
                    all_coords = coords
                    all_rgb = rgb
                num_pts = all_coords.shape[0]
                # print('all_coords',all_coords)
                xyz = all_coords * 0.6
                rgb = all_rgb
            
            else:
                thetas = np.random.rand(num_pts)*np.pi
                phis = np.random.rand(num_pts)*2*np.pi        
                radius = np.random.rand(num_pts)*0.05
                # We create random points inside the bounds of sphere
                xyz_ball = np.stack([
                    radius * np.sin(thetas) * np.sin(phis),
                    radius * np.sin(thetas) * np.cos(phis),
                    radius * np.cos(thetas),
                ], axis=-1) # [B, 3]expend_dims
                rgb_ball = np.random.random((50000, num_pts, 3))*0.0001
                rgb = (np.expand_dims(rgb,axis=1)+rgb_ball).reshape(-1,3)
                xyz = (np.expand_dims(xyz,axis=1)+np.expand_dims(xyz_ball,axis=0)).reshape(-1,3)
                xyz = xyz * 1.
                num_pts = xyz.shape[0]
        else:
            raise NotImplementedError()
            print(f"Generating random point cloud ({num_pts})...")

        shs = np.random.random((num_pts, 3)) / 255.0

        # print('xyz',xyz)
        if opt.use_pointe_rgb:
            # breakpoint()
            pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros((num_pts, 3)))
            storePly(ply_path, xyz, rgb * 255)
        else:
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # print('aaa')

    scene_info = RSceneInfo(point_cloud=pcd,
                           test_cameras=test_cam_infos,
                           ply_path=ply_path)
    return scene_info
#borrow from https://github.com/ashawkey/stable-dreamfusion

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

# def circle_poses(radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), angle_overhead=30, angle_front=60):

#     theta = theta / 180 * np.pi
#     phi = phi / 180 * np.pi
#     angle_overhead = angle_overhead / 180 * np.pi
#     angle_front = angle_front / 180 * np.pi

#     centers = torch.stack([
#         radius * torch.sin(theta) * torch.sin(phi),
#         radius * torch.cos(theta),
#         radius * torch.sin(theta) * torch.cos(phi),
#     ], dim=-1) # [B, 3]

#     # lookat
#     forward_vector = safe_normalize(centers)
#     up_vector = torch.FloatTensor([0, 1, 0]).unsqueeze(0).repeat(len(centers), 1)
#     right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
#     up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

#     poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(centers), 1, 1)
#     poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
#     poses[:, :3, 3] = centers

#     return poses.numpy()

def circle_poses(radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), angle_overhead=30, angle_front=60):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.sin(theta) * torch.cos(phi),
        radius * torch.cos(theta),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers
    # print('poses')
    return poses.numpy()

def certain_poses(opt, size, radius_range=[1, 1.5], theta=torch.tensor([90]), phi=torch.tensor([0]), angle_overhead=30, angle_front=60):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi
    radius = gen_random_pos(size, radius_range)
    

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.sin(theta) * torch.cos(phi),
        radius * torch.cos(theta),
    ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if opt.jitter_pose:
        jit_center = opt.jitter_center # 0.015  # was 0.2
        jit_target = opt.jitter_target
        centers += torch.rand_like(centers) * jit_center - jit_center/2.0
        targets += torch.randn_like(centers) * jit_target
    
    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    
    if opt.jitter_pose:
        up_noise = torch.randn_like(up_vector) * opt.jitter_up
    else:
        up_noise = 0
    
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses.numpy(), radius.numpy()

def gen_random_pos(size, param_range, gamma=1):
    lower, higher = param_range[0], param_range[1]
    
    mid = lower + (higher - lower) * 0.5
    radius = (higher - lower) * 0.5

    rand_ = torch.rand(size) # 0, 1
    sign = torch.where(torch.rand(size) > 0.5, torch.ones(size) * -1., torch.ones(size))
    rand_ = sign * (rand_ ** gamma)          

    return (rand_ * radius) + mid


def rand_poses(size, opt, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], angle_overhead=30, angle_front=60, uniform_sphere_rate=0.5, rand_cam_gamma=1):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    # radius = torch.rand(size) * (radius_range[1] - radius_range[0]) + radius_range[0]
    radius = gen_random_pos(size, radius_range)

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                torch.randn(size),
                torch.abs(torch.randn(size)),
                torch.randn(size),
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        # thetas = torch.rand(size) * (theta_range[1] - theta_range[0]) + theta_range[0]
        # phis = torch.rand(size) * (phi_range[1] - phi_range[0]) + phi_range[0]
        # phis[phis < 0] += 2 * np.pi

        # centers = torch.stack([
        #     radius * torch.sin(thetas) * torch.sin(phis),
        #     radius * torch.cos(thetas),
        #     radius * torch.sin(thetas) * torch.cos(phis),
        # ], dim=-1) # [B, 3]
        # thetas = torch.rand(size) * (theta_range[1] - theta_range[0]) + theta_range[0]
        # phis = torch.rand(size) * (phi_range[1] - phi_range[0]) + phi_range[0]
        thetas = gen_random_pos(size, theta_range, rand_cam_gamma)
        phis = gen_random_pos(size, phi_range, rand_cam_gamma)
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.sin(thetas) * torch.cos(phis),
            radius * torch.cos(thetas),
        ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if opt.jitter_pose:
        jit_center = opt.jitter_center # 0.015  # was 0.2
        jit_target = opt.jitter_target
        centers += torch.rand_like(centers) * jit_center - jit_center/2.0
        targets += torch.randn_like(centers) * jit_target

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
    #up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if opt.jitter_pose:
        up_noise = torch.randn_like(up_vector) * opt.jitter_up
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise) #forward_vector

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1) #up_vector
    poses[:, :3, 3] = centers


    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses.numpy(), thetas.numpy(), phis.numpy(), radius.numpy()

def GenerateCircleCameras(opt, size=8, render45 = False):
    # random focal
    fov = opt.default_fovy
    cam_infos = []
    #generate specific data structure
    for idx in range(size):
        thetas = torch.FloatTensor([opt.default_polar])
        phis = torch.FloatTensor([(idx / size) * 360])
        radius = torch.FloatTensor([opt.default_radius])
        # random pose on the fly
        poses = circle_poses(radius=radius, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
        # print(poses[0])
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,width=opt.image_w, 
                        height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius,
                        pose = poses))  
    if render45:
        for idx in range(size):
            thetas = torch.FloatTensor([opt.default_polar*2//3])
            phis = torch.FloatTensor([(idx / size) * 360])
            radius = torch.FloatTensor([opt.default_radius])
            # random pose on the fly
            poses = circle_poses(radius=radius, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]
            fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
            FovY = fovy
            FovX = fov

            # delta polar/azimuth/radius to default view
            delta_polar = thetas - opt.default_polar
            delta_azimuth = phis - opt.default_azimuth
            delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
            delta_radius = radius - opt.default_radius
            cam_infos.append(RandCameraInfo(uid=idx+size, R=R, T=T, FovY=FovY, FovX=FovX,width=opt.image_w, 
                            height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius, pose = poses))         
    return cam_infos


def GenerateCertainCameras(opt, size=8, SSAA=True, azimuth = 0, polar = 90):
    # random focal
    fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]
    cam_infos = []
    #generate specific data structure
    thetas = torch.FloatTensor([polar])
    phis = torch.FloatTensor([azimuth])
    # random pose on the fly
    poses, radius = certain_poses(opt, size, radius_range=opt.radius_range, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
    
    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1
        
    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa
    
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    
    for idx in range(size):
        matrix = np.linalg.inv(poses[idx])
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, image_h), image_w)
        FovY = fovy
        FovX = fov

        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,width=image_w, 
                        height = image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius[idx],
                        pose = poses[idx]))
    return cam_infos


def GenerateRandomCameras(opt, size=2000, SSAA=True):
    # random pose on the fly
    poses, thetas, phis, radius = rand_poses(size, opt, radius_range=opt.radius_range, theta_range=opt.theta_range, phi_range=opt.phi_range, 
                                             angle_overhead=opt.angle_overhead, angle_front=opt.angle_front, uniform_sphere_rate=opt.uniform_sphere_rate,
                                             rand_cam_gamma=opt.rand_cam_gamma)
    # print('thetas',thetas)
    # print('phis',phis)
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # print('delta_polar_aaaaa',delta_polar)
    # random focal
    fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]
    
    cam_infos = []

    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    #generate specific data structure
    for idx in range(size):
        matrix = np.linalg.inv(poses[idx])
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        # matrix = poses[idx]
        # R = matrix[:3,:3]
        # T = matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, image_h), image_w)
        FovY = fovy
        FovX = fov

        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,width=image_w, 
                                        height=image_h, delta_polar = delta_polar[idx],
                                        delta_azimuth = delta_azimuth[idx], delta_radius = delta_radius[idx], pose = poses[idx]))           
    return cam_infos


def GeneratePurnCameras(opt, size=300):
    # random pose on the fly
    poses, thetas, phis, radius = rand_poses(size, opt, radius_range=[opt.default_radius,opt.default_radius+0.1], theta_range=opt.theta_range, phi_range=opt.phi_range, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front, uniform_sphere_rate=opt.uniform_sphere_rate)
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # random focal
    #fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]
    fov = opt.default_fovy
    cam_infos = []
    #generate specific data structure
    for idx in range(size):
        matrix = np.linalg.inv(poses[idx])     
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        # matrix = poses[idx]
        # R = matrix[:3,:3]
        # T = matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,width=opt.image_w, 
                        height = opt.image_h, delta_polar = delta_polar[idx],delta_azimuth = delta_azimuth[idx], delta_radius = delta_radius[idx]))           
    return cam_infos

sceneLoadTypeCallbacks = {
    # "Colmap": readColmapSceneInfo,
    # "Blender" : readNerfSyntheticInfo,
    "RandomCam" : readCircleCamInfo
}