import os
from tqdm import tqdm
from random import randint
import numpy as np
import random
from collections import OrderedDict
import torch
from PIL import Image
from einops import rearrange
import scipy
import imageio
import glob
import cv2
from lietorch import SE3

import open3d as o3d

from scene.gaussian_model_ht import CF3DGS_Render as GS_Render

from utils.graphics_utils import BasicPointCloud
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr, colorize
from utils.utils_poses.align_traj import align_ate_c2b_use_a2b
from utils.utils_poses.comp_ate import compute_rpe, compute_ATE

from kornia.geometry.depth import depth_to_3d

import pdb

from .trainer import GaussianTrainer
from .losses import Loss

from copy import copy
from utils.vis_utils import interp_poses_bspline, plot_pose

from scipy.spatial.transform import Rotation as R


class HTGaussianTrainer(GaussianTrainer):
    def __init__(self, data_root, model_cfg, pipe_cfg, optim_cfg):
        super().__init__(data_root, model_cfg, pipe_cfg, optim_cfg)
        self.model_cfg = model_cfg
        self.pipe_cfg = pipe_cfg
        self.optim_cfg = optim_cfg

        self.gs_render = GS_Render(white_background=False,
                                   view_dependent=model_cfg.view_dependent,)
        self.gs_render_local = GS_Render(white_background=False,
                                         view_dependent=model_cfg.view_dependent,)

        self.gs_render_local_2 = GS_Render(white_background=False,
                                         view_dependent=model_cfg.view_dependent,) # Used for VFI

        if self.pipe_cfg.train_mode in ['hierarchical_training', 'progressive_training']:
            if 'base' in self.pipe_cfg.multi_source_supervision:
                self.train_level = self.pipe_cfg.train_level
                self.gs_render_list = [[self.gs_render]]
                self.to_visit_frames_dict = {}
                for level_curr in range(1, self.train_level + 1): # Level 0 is stored in self.gs_render
                    gs_render_list = []
                    for i in range(2 ** level_curr):
                        gs_render_list.append(GS_Render(white_background=False, view_dependent=model_cfg.view_dependent))
                    self.gs_render_list.append(gs_render_list)
            else:
                self.train_level = self.pipe_cfg.train_level
                self.gs_render_list = []
                self.to_visit_frames_dict = {}
                for i in range(2 ** self.train_level):
                    self.gs_render_list.append(GS_Render(white_background=False, view_dependent=model_cfg.view_dependent))

        self.use_mask = self.pipe_cfg.use_mask
        self.use_mono = self.pipe_cfg.use_mono
        self.near = 0.01
        self.setup_losses()


    def setup_losses(self):
        self.loss_func = Loss(self.optim_cfg)

    def train_step(self,
                   gs_render,
                   viewpoint_cam,
                   iteration,
                   pipe,
                   optim_opt,
                   colors_precomp=None,
                   update_gaussians=True,
                   update_cam=True,
                   update_distort=False,
                   densify=True,
                   prev_gaussians=None,
                   use_reproject=False,
                   use_matcher=False,
                   ref_fidx=None,
                   reset=True,
                   reproj_loss=None,
                   mask=None,
                   **kwargs,
                   ):
        # Render
        render_pkg = gs_render.render(
            viewpoint_cam,
            compute_cov3D_python=pipe.compute_cov3D_python,
            convert_SHs_python=pipe.convert_SHs_python,
            override_color=colors_precomp)

        if prev_gaussians is not None:
            with torch.no_grad():
                # Render
                render_pkg_prev = prev_gaussians.render(
                    viewpoint_cam,
                    compute_cov3D_python=pipe.compute_cov3D_python,
                    convert_SHs_python=pipe.convert_SHs_python,
                    override_color=colors_precomp)
            mask = (render_pkg["alpha"] > 0.5).float()
            render_pkg["image"] = render_pkg["image"] * \
                mask + render_pkg_prev["image"] * (1 - mask)
            render_pkg["depth"] = render_pkg["depth"] * \
                mask + render_pkg_prev["depth"] * (1 - mask)

        image, viewspace_point_tensor, visibility_filter, radii = (render_pkg["image"],
                                                                   render_pkg["viewspace_points"],
                                                                   render_pkg["visibility_filter"],
                                                                   render_pkg["radii"])
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        loss_dict = self.compute_loss(render_pkg, viewpoint_cam,
                                      pipe, iteration,
                                      use_reproject, use_matcher,
                                      ref_fidx, **kwargs)

        loss = loss_dict['loss']
        loss.backward()

        with torch.no_grad():
            psnr_train = psnr(image, gt_image).mean().double()
            self.just_reset = False
            if iteration < optim_opt.densify_until_iter and densify:
                # Keep track of max radii in image-space for pruning
                try:
                    gs_render.gaussians.max_radii2D[visibility_filter] = torch.max(gs_render.gaussians.max_radii2D[visibility_filter],
                                                                                   radii[visibility_filter])
                except:
                    pdb.set_trace()
                gs_render.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter)

                if iteration > optim_opt.densify_from_iter and iteration % optim_opt.densification_interval == 0:
                    size_threshold = 20 if iteration > optim_opt.opacity_reset_interval else None
                    gs_render.gaussians.densify_and_prune(optim_opt.densify_grad_threshold, 0.005,
                                                               gs_render.radius, size_threshold)

                if iteration % optim_opt.opacity_reset_interval == 0 and reset and iteration < optim_opt.reset_until_iter:
                    gs_render.gaussians.reset_opacity()
                    self.just_reset = True

            if update_gaussians:
                gs_render.gaussians.optimizer.step()
                gs_render.gaussians.optimizer.zero_grad(set_to_none=True)
            if getattr(gs_render.gaussians, "camera_optimizer", None) is not None and update_cam:
                current_fidx = gs_render.gaussians.seq_idx
                gs_render.gaussians.camera_optimizer[current_fidx].step()
                gs_render.gaussians.camera_optimizer[current_fidx].zero_grad(
                    set_to_none=True)


        return loss_dict, render_pkg, psnr_train

    def init_leaf_3DGS(self, view_idx, pipe, optim_opt, gs_render=None):
        """
        Initialize the base 3DGS model by training on a single view.

        This is used for the first frame of a sequence.

        Args:
            view_idx: The index of the view to train on.
            pipe: The data processing pipeline.
            optim_opt: The optimization options.
            gs_render: The GaussianRender object to use. If None, use self.gs_render.
        """
        if gs_render is None:
            gs_render = self.gs_render

        # prepare data
        self.loss_func.depth_loss_type = "invariant"
        _, pcd, viewpoint_cam = self.prepare_data(view_idx,
                                                         orthogonal=True,
                                                         down_sample=True)

        # Initialize gaussians
        gs_render.reset_model()
        gs_render.init_model(pcd,)
        gs_render.gaussians.init_RT_seq(self.seq_len) # initialize the RT sequence 
        gs_render.gaussians.set_seq_idx(view_idx)
        gs_render.gaussians.rotate_seq = False
        optim_opt.iterations = 1000
        optim_opt.densify_from_iter = optim_opt.iterations + 1
        gs_render.gaussians.training_setup(optim_opt, fix_pos=True,)
        for iteration in range(1, optim_opt.iterations+1): # Orris: default: 30_000
            gs_render.gaussians.update_learning_rate(iteration)
            loss, rend_dict, psnr_train = self.train_step(gs_render,
                                                          viewpoint_cam, iteration,
                                                          pipe, optim_opt,
                                                          depth_gt=self.mono_depth[view_idx],
                                                          update_gaussians=True,
                                                          update_cam=False,
                                                          )
            if iteration % 10 == 0:
                loss_display = {k_: v_.item() for k_, v_ in loss.items()}
                self.logger.info(f"[init_leaf_3DGS] iter {iteration} || psnr_train: {psnr_train} loss: {loss_display}")

    def merge_two_3DGS(self, gs_render_dst, gs_render_src, transform_matrix, 
                        to_visit_frames_dst=None, to_visit_frames_src=None):
        """
        Merge two 3DGS models.

        Args:
            gs_render_dst: The destination GaussianRender object to use.
            gs_render_src: The source GaussianRender object to use.
            transform_matrix: The 4x4 transformation matrix from the source to the destination.
            to_visit_frames_dst: The training frames used for the destination 3DGS
            to_visit_frames_src: The training frames used for the source 3DGS
        """
        self.logger.info(f"The number of 3D Gaussians in the first 3DGS before merging: {gs_render_dst.gaussians.get_xyz.shape[0]}" )
        self.logger.info(f"The number of 3D Gaussians in the second 3DGS before merging: {gs_render_src.gaussians.get_xyz.shape[0]}" )
        cameras_dst = []
        for fidx in to_visit_frames_dst:
            pose = gs_render_dst.gaussians.get_RT(fidx).detach().cpu() if not gs_render_dst.gaussians.rotate_seq else None
            viewpoint_cam = self.load_viewpoint_cam(fidx, pose=pose, load_depth=True)
            cameras_dst.append(viewpoint_cam)
        color_importance_dst = HTGaussianTrainer.calc_importance(gs_render_dst, cameras_dst, self.pipe_cfg)
        color_importance_dst = color_importance_dst.amax(-1).squeeze()
        top_values_dst, top_indices_dst = torch.topk(color_importance_dst, int(color_importance_dst.shape[0] * self.pipe_cfg.prune_ratio), largest=False)

        mask_dst = torch.zeros((gs_render_dst.gaussians._xyz.shape[0], 1)).cuda()
        mask_dst[top_indices_dst, :] = 1
        mask_dst = mask_dst.bool()

        gs_render_dst.gaussians.prune_points(mask_dst.squeeze())
        cameras_src = []
        for fidx in to_visit_frames_src:
            pose = gs_render_src.gaussians.get_RT(fidx).detach().cpu() if not gs_render_src.gaussians.rotate_seq else None
            viewpoint_cam = self.load_viewpoint_cam(fidx, pose=pose, load_depth=True)
            cameras_src.append(viewpoint_cam)
        color_importance_src = HTGaussianTrainer.calc_importance(gs_render_src, cameras_src, self.pipe_cfg)
        color_importance_src = color_importance_src.amax(-1).squeeze()
        top_values_src, top_indices_src = torch.topk(color_importance_src, int(color_importance_src.shape[0] * self.pipe_cfg.prune_ratio), largest=False)

        mask_src = torch.zeros((gs_render_src.gaussians._xyz.shape[0], 1)).cuda()
        mask_src[top_indices_src, :] = 1
        mask_src = mask_src.bool()

        # transform points
        n = len(gs_render_src.gaussians._xyz)
        points_homogeneous = torch.cat([gs_render_src.gaussians._xyz, torch.ones((n, 1)).cuda()], dim=1) 
        aligned_xyz_homogeneous = points_homogeneous @ transform_matrix.T
        aligned_xyz = aligned_xyz_homogeneous[:, :3] / aligned_xyz_homogeneous[:, 3].unsqueeze(1)

        valid_points_mask = ~(mask_src.squeeze())
        new_xyz = aligned_xyz[valid_points_mask]
        new_features_dc = gs_render_src.gaussians._features_dc[valid_points_mask]
        new_features_rest = gs_render_src.gaussians._features_rest[valid_points_mask]
        new_opacity = gs_render_src.gaussians._opacity[valid_points_mask]
        new_scaling = gs_render_src.gaussians._scaling[valid_points_mask]
        new_rotation = gs_render_src.gaussians._rotation[valid_points_mask]


        gs_render_dst.gaussians.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        self.logger.info(f"The number of 3D Gaussians in the merged 3DGS: {gs_render_dst.gaussians.get_xyz.shape[0]}")

    def train_single_image_3DGS(self, optim_opt, viewpoint_cam, pipe, gs_render):
        """
            Train the 3DGS model in gs_render using a single view.
            This method is different from "self.train_relative_pose", because the optimized parameters are the attributes of 3D Gaussians.

            Args:
                optim_opt: The optimization options.
                viewpoint_cam: The viewpoint camera to use for training.
                pipe: The configuration.
                gs_render (optional): The GaussianRender object to use.
        """
        # Fit current gaussian
        gs_render.gaussians.training_setup(optim_opt, fix_pos=True,)

        for iteration in range(1, optim_opt.iterations+1):
            # Update learning rate
            gs_render.gaussians.update_learning_rate(iteration)
            loss, rend_dict, psnr_train = self.train_step(gs_render,
                                                          viewpoint_cam, iteration, # Train local 3DGS initialized from the previous frame using the previous frame
                                                          pipe, optim_opt,
                                                          update_gaussians=True,
                                                          update_cam=False,
                                                          updata_distort=False,
                                                          densify=False,
                                                          )
            if psnr_train > 35 and iteration > 500:
                break

            if iteration % 10 == 0:
                loss_display = {k_: v_.item() for k_, v_ in loss.items()}
                self.logger.info(f"[Train single-image 3DGS] iter {iteration} || PSNR: {psnr_train} Loss: {loss_display} Number of 3D Gaussians: {gs_render.gaussians.get_xyz.shape[0]}")


    def train_relative_pose(self, gs_render, optim_opt, viewpoint_cam, pipe):
        """
            Train relative pose between two views.
            This method is different from "self.train_single_image_3DGS", because the optimized parameters are the relative poses.
            
            Args:
                gs_render: the renderer
                optim_opt: optimization options
                viewpoint_cam: the viewpoint camera
                pipe: the pipeline options
            Returns:
                loss: the loss after training
        """
        gs_render.gaussians.training_setup_fix_position(optim_opt, gaussian_rot=False)

        for iteration in range(1, optim_opt.iterations+1):
            # Update learning rate
            loss, rend_dict_ref, psnr_train = self.train_step(gs_render,
                                                            viewpoint_cam, iteration, 
                                                            pipe, optim_opt,
                                                            densify=False,
                                                            )
            if iteration % 10 == 0:
                loss_display = {k_: v_.item() for k_, v_ in loss.items()}
                self.logger.info(f"[Train relative pose] iter {iteration} || PSNR: {psnr_train} Loss: {loss_display} Number of 3D Gaussians: {gs_render.gaussians.get_xyz.shape[0]}")

        return loss


    def compute_relative_pose(self, view_idx, view_idx_prev):
        """
            Compute the relative pose between two views.
            The optimized relative pose is saved in "self.pose_dict['rel_pose_{view_idx_prev}_to_{view_idx}']"

            Args:
                view_idx: The view index for the current frame.
                view_idx_prev: The view index for the previous frame.

            Returns:
                res_dict: A dictionary containing the loss after training.
        """
        if self.pipe_cfg.train_pose_mode == 'vfi':
            return self.compute_relative_pose_vfi(view_idx, view_idx_prev)
        if f'rel_pose_{view_idx_prev}_to_{view_idx}' in self.pose_dict:
            return # no need to run this code

        # Initialize gaussians
        self.loss_func.depth_loss_type = "invariant"
        pipe = copy(self.pipe_cfg)
        optim_opt = copy(self.optim_cfg)
        optim_opt.iterations = 1000
        optim_opt.densify_from_iter = optim_opt.iterations + 1

        _, pcd, viewpoint_cam = self.prepare_data(view_idx_prev, orthogonal=True, down_sample=True)
        # Initialize the Local 3DGS
        self.gs_render_local.reset_model()
        self.gs_render_local.init_model(pcd) # Initialize from the point cloud of the previous frame

        self.logger.info(f"Train Frame {view_idx_prev}")
        self.train_single_image_3DGS(optim_opt, viewpoint_cam, pipe, gs_render=self.gs_render_local)

        viewpoint_cam_ref = self.load_viewpoint_cam(view_idx, load_depth=True)
        optim_opt.iterations = 300
        optim_opt.densify_from_iter = optim_opt.iterations + 1

        self.gs_render_local.gaussians.init_RT(None)

        # view_idx_prev --> view_idx
        res_dict = {}
        loss_train_relative_pose = self.train_relative_pose(self.gs_render_local, optim_opt, viewpoint_cam_ref, pipe)
        res_dict['loss_train_relative_pose'] = loss_train_relative_pose

        rel_pose = self.gs_render_local.gaussians.get_RT().detach()
        self.pose_dict[f'rel_pose_{view_idx_prev}_to_{view_idx}'] = rel_pose

        return res_dict

    def compute_relative_pose_vfi(self, view_idx, view_idx_prev):
        """
            Similar to "self.compute_relative_pose", but we use interpolated frames
        """
        if f'rel_pose_{view_idx_prev}_to_{view_idx}' in self.pose_dict:
            return # no need to run this code

        # Initialize gaussians
        self.loss_func.depth_loss_type = "invariant"
        pipe = copy(self.pipe_cfg)
        optim_opt = copy(self.optim_cfg)
        optim_opt.iterations = 1000
        optim_opt.densify_from_iter = optim_opt.iterations + 1

        _, pcd, viewpoint_cam, (pcd_vfi, viewpoint_cam_vfi) = self.prepare_data(view_idx_prev, orthogonal=True, down_sample=True, load_vfi=True)
        # Initialize the Local 3DGS
        self.gs_render_local.reset_model()
        self.gs_render_local.init_model(pcd) # Initialize from the point cloud of the previous frame
        self.logger.info(f"Train Frame {view_idx_prev}")
        self.train_single_image_3DGS(optim_opt, viewpoint_cam, pipe, gs_render=self.gs_render_local)

        self.gs_render_local_2.reset_model()
        self.gs_render_local_2.init_model(pcd_vfi) # Initialize from the point cloud of the previous frame
        self.logger.info(f"Train Frame {view_idx_prev+0.5}")
        self.train_single_image_3DGS(optim_opt, viewpoint_cam_vfi, pipe, gs_render=self.gs_render_local_2)

        viewpoint_cam_ref = self.load_viewpoint_cam(view_idx, load_depth=True, load_vfi=False) # We do not load this one, only the previous index
        optim_opt.iterations = 300
        optim_opt.densify_from_iter = optim_opt.iterations + 1

        self.gs_render_local.gaussians.init_RT(None)
        self.gs_render_local_2.gaussians.init_RT(None)

        # view_idx_prev --> view_idx
        res_dict = {}
        loss_train_relative_pose_1 = self.train_relative_pose(self.gs_render_local, optim_opt, viewpoint_cam_vfi, pipe)
        loss_train_relative_pose_2 = self.train_relative_pose(self.gs_render_local_2, optim_opt, viewpoint_cam_ref, pipe)
        res_dict['loss_train_relative_pose_1'] = loss_train_relative_pose_1
        res_dict['loss_train_relative_pose_2'] = loss_train_relative_pose_2

        rel_pose1 = self.gs_render_local.gaussians.get_RT().detach()
        rel_pose2 = self.gs_render_local_2.gaussians.get_RT().detach()
        rel_pose = rel_pose2 @ rel_pose1
        self.pose_dict[f'rel_pose_{view_idx_prev}_to_{view_idx_prev}.5'] = rel_pose1
        self.pose_dict[f'rel_pose_{view_idx_prev}.5_to_{view_idx}'] = rel_pose2
        self.pose_dict[f'rel_pose_{view_idx_prev}_to_{view_idx}'] = rel_pose

        return res_dict



    def render_frame(self, view_idx, gs_render):
        """
            Given a frame index, render the frame and evaluate the PSNR of the rendered image with respect to the ground truth image.

            Parameters
            ----------
            view_idx (int): The index of the frame to render.
            gs_render (CF3DGS_Render): The CF3DGS_Render object to use for rendering.

            Returns
            -------
            psnr_train (float): The PSNR between the rendered image and the ground truth image.
            render_dict (dict): The dictionary containing the rendered image and other information.
            gt_image (torch.Tensor): The ground truth image.
        """
        with torch.no_grad():
            pose = gs_render.gaussians.get_RT(view_idx).detach().cpu() if not gs_render.gaussians.rotate_seq else None
            viewpoint_cam = self.load_viewpoint_cam(view_idx,
                                                    pose=pose,
                                                    load_depth=True)
            render_dict = gs_render.render(viewpoint_cam,
                                                compute_cov3D_python=self.pipe_cfg.compute_cov3D_python,
                                                convert_SHs_python=self.pipe_cfg.convert_SHs_python)
            gt_image = viewpoint_cam.original_image.cuda()
            psnr_train = psnr(render_dict["image"], gt_image).mean().double()
        return psnr_train, render_dict, gt_image

    def get_virtual_view(self, pose0, pose1, alpha):
        """
            Computes an interpolated pose between two given poses based on a blending factor.

            Args:
                pose0: The initial pose as a transformation matrix.
                pose1: The target pose as a transformation matrix.
                alpha: A blending factor between 0 and 1, inclusive, where 0 returns pose0 and 1 returns pose1.

            Returns:
                torch.Tensor: The interpolated pose as a transformation matrix.

            Note:
                This function uses SE3 logarithm and exponential maps to perform the interpolation.
        """
        assert 0 <= alpha and alpha <= 1
        pose = (pose0 * SE3.exp(SE3.log((pose0.inv() * pose1)) * alpha)).matrix().detach()
        return pose.squeeze()


    def sample_a_training_frame(self, view_idx):
        """
            Randomly samples a frame index from the frames visited so far.

            The sampling strategy is as follows: with probability 0.7, the sampled frame is from the second half of the visited frames, and with probability 0.3, the sampled frame is from the first half.

            Args:
                view_idx (int): The index of the current frame.

            Returns:
                int: The index of the sampled frame.
        """
        if hasattr(self, 'visited_frames'):
            last_idx = max(1, len(self.visited_frames)//2) 
            if random.random() < 0.7:
                idx = randint(last_idx, len(self.visited_frames) - 1)
            else:
                idx = randint(1, last_idx)
            fidx = self.visited_frames[idx]
        else:
            last_frame = max(1, view_idx//2)
            if random.random() < 0.7:
                fidx = randint(last_frame, view_idx)
            else:
                fidx = randint(1, last_frame)
        return fidx


    def train_leaf_3DGS(self, view_idx, view_idx_prev, gs_render):
        """
            Train the leaf 3DGS model on a sequence of frames.

            This function performs training of the leaf 3DGS model by iterating over a 
            sequence of frames. 

            Args:
                view_idx (int): The index of the current view/frame being trained on.
                view_idx_prev (int): The index of the previous view/frame.
                gs_render (GaussianRender): The GaussianRender object used for rendering.
                                            If None, defaults to self.gs_render.
        """

        pipe = copy(self.pipe_cfg)

        gs_render.gaussians.rotate_seq = False
        pipe.convert_SHs_python = gs_render.gaussians.rotate_seq

        optim_cfg = copy(self.optim_cfg)
        optim_cfg.densification_interval = optim_cfg.densification_interval_leaf

        if self.just_reset:
            num_iterations = 500
            self.just_reset = False
            for iteration in range(1, num_iterations):
                fidx = randint(0, view_idx_prev) # Randomly load one of the saved frame
                self.global_iteration += 1
                gs_render.gaussians.update_learning_rate(self.global_iteration)
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=gs_render.gaussians.get_RT(
                                                            fidx).detach().cpu(),
                                                        load_depth=True)
                loss, rend_dict_ref, psnr_train = self.train_step(gs_render,
                                                                  viewpoint_cam,
                                                                  self.global_iteration,
                                                                  pipe, optim_cfg,
                                                                  update_gaussians=True,
                                                                  update_cam=False,
                                                                  update_distort=False,
                                                                  )


        num_iterations = self.single_step
        
        for iteration in range(1, num_iterations+1):
            fidx = self.sample_a_training_frame(view_idx)
            self.global_iteration += 1

            pose_train = gs_render.gaussians.get_RT(fidx).detach().cpu() if not gs_render.gaussians.rotate_seq else None
            if (fidx + 1 < self.seq_len) and ('vfi' in self.pipe_cfg.multi_source_supervision) and (random.random() < optim_cfg.mss_phase2_ratio):
                pose_train = self.pose_dict[f'rel_pose_{fidx}_to_{fidx+0.5}'].detach().cpu() @ pose_train # Adjust the pose
                viewpoint_cam = self.load_viewpoint_cam(fidx, pose=pose_train, load_depth=True, load_vfi=True) # wrt the 1st 3DGS
                viewpoint_cam.original_image = viewpoint_cam.vfi
            else:
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=pose_train,
                                                        load_depth=True)

            gs_render.gaussians.update_learning_rate(self.global_iteration)

            loss, rend_dict_ref, psnr_train = self.train_step(gs_render,
                                                              viewpoint_cam,
                                                              self.global_iteration,
                                                              pipe, optim_cfg,
                                                              update_gaussians=True,
                                                              update_cam=False,
                                                              update_distort=self.pipe_cfg.distortion,
                                                              )

            if self.global_iteration % 1000 == 0:
                gs_render.gaussians.oneupSHdegree()

            if iteration % 10 == 0:
                loss_display = {k_: v_.item() for k_, v_ in loss.items()}
                self.logger.info(f"[Train leaf 3DGS] global iter {self.global_iteration} || iter {iteration} || PSNR: {psnr_train} Loss: {loss_display} Number of 3D Gaussians: {gs_render.gaussians.get_xyz.shape[0]}")


    def train_nonleaf_3DGS_phase2(self, indices, num_iterations, gs_render):
        """
            Train the non-leaf 3D Gaussian Splatting (3DGS) model in phase 2.

            Supervision: original training images + interpolated frames from video frame interpolation models

            Args:
                gs_render (GaussianRender): The GaussianRender object used for rendering
                    and training.
                gs_render_children (list of GaussianRender): A list of child GaussianRender
                    objects whose frames are considered during training.
        """
        pipe = copy(self.pipe_cfg)

        gs_render.gaussians.rotate_seq = False
        pipe.convert_SHs_python = gs_render.gaussians.rotate_seq 

        optim_cfg = copy(self.optim_cfg)
        optim_cfg.densification_interval = optim_cfg.mss_phase2_densification_interval

        if optim_cfg.mss_phase2_densify_until_iter_ratio is not None:
            optim_cfg.densify_until_iter = int(num_iterations * optim_cfg.mss_phase2_densify_until_iter_ratio)

        for iteration in range(1, num_iterations+1):
            fidx = random.choice(indices)
            self.global_iteration += 1

            pose_train = gs_render.gaussians.get_RT(fidx).detach().cpu() if not gs_render.gaussians.rotate_seq else None
            if (fidx + 1 < self.seq_len) and ('vfi' in self.pipe_cfg.multi_source_supervision) and (random.random() < optim_cfg.mss_phase2_ratio):
                pose_train = self.pose_dict[f'rel_pose_{fidx}_to_{fidx+0.5}'].detach().cpu() @ pose_train # Adjust the pose
                viewpoint_cam = self.load_viewpoint_cam(fidx, pose=pose_train, load_depth=True, load_vfi=True) # wrt the 1st 3DGS
                viewpoint_cam.original_image = viewpoint_cam.vfi
            else:
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=pose_train,
                                                        load_depth=True)
            # Update learning rate
            gs_render.gaussians.update_learning_rate(self.global_iteration)

            loss, rend_dict_ref, psnr_train = self.train_step(gs_render,
                                                              viewpoint_cam,
                                                              self.global_iteration,
                                                              pipe, optim_cfg,
                                                              update_gaussians=True,
                                                              update_cam=False,
                                                              update_distort=self.pipe_cfg.distortion,
                                                              )

            if self.global_iteration % 1000 == 0:
                gs_render.gaussians.oneupSHdegree()

            if iteration % 100 == 0:
                loss_display = {k_: v_.item() for k_, v_ in loss.items()}
                self.logger.info(f"[Train nonleaf 3DGS] global iter {self.global_iteration} || iter {iteration} || PSNR: {psnr_train} Loss: {loss_display} Number of 3D Gaussians: {gs_render.gaussians.get_xyz.shape[0]}")



    def create_pcd_from_render(self, render_dict, viewpoint_cam):
        intrinsics = torch.from_numpy(viewpoint_cam.intrinsics).float().cuda()
        depth = render_dict["depth"].squeeze()
        image = render_dict["image"]
        pts = depth_to_3d(depth[None, None],
                          intrinsics[None],
                          normalize_points=False)
        points = pts.squeeze().permute(1, 2, 0).detach().cpu().reshape(-1, 3).numpy()
        colors = image.permute(1, 2, 0).detach().cpu().reshape(-1, 3).numpy()
        pcd_data = o3d.geometry.PointCloud()
        pcd_data.points = o3d.utility.Vector3dVector(points)
        pcd_data.colors = o3d.utility.Vector3dVector(colors)
        pcd_data = pcd_data.farthest_point_down_sample(num_samples=30_000)
        colors = np.asarray(pcd_data.colors, dtype=np.float32)
        points = np.asarray(pcd_data.points, dtype=np.float32)
        normals = np.asarray(pcd_data.normals, dtype=np.float32)
        pcd = BasicPointCloud(points, colors, normals)
        return pcd

    def hierarchical_training(self):
        """
            Hierarchical training of 3DGS. 
            
            First, the relative poses are learned between all the frames.
            Then, starting from the highest level, each pair of adjacent segments are merged and the merged segment is fine-tuned.
            The process is repeated until the lowest level is reached.
        """
        pipe = copy(self.pipe_cfg)
        self.single_step = self.optim_cfg.single_step

        num_iterations = self.single_step * (self.seq_len // 10) * 10
        self.optim_cfg.iterations = num_iterations
        self.optim_cfg.position_lr_max_steps = num_iterations
        self.optim_cfg.opacity_reset_interval = num_iterations // 10
        self.optim_cfg.densify_until_iter = num_iterations
        self.optim_cfg.reset_until_iter = int(num_iterations * 0.8)
        self.optim_cfg.densify_from_iter = self.single_step

        pipe.convert_SHs_python = True

        self.pose_dict = dict()

        os.makedirs(f"{self.result_path}/chkpnt", exist_ok=True)
        os.makedirs(f"{self.result_path}/pose", exist_ok=True)

        if self.pipe_cfg.load_pose is not None:
            self.logger.info(f"We load {self.pipe_cfg.load_pose}")
            self.pose_dict = torch.load(self.pipe_cfg.load_pose)
        else:
            self.logger.info(f"We do not load poses")
            self.pose_dict = {}

        for fidx in range(1, self.seq_len):
            self.compute_relative_pose(fidx, fidx-1) # rel_pose: view_idx_prev --> view_idx

        lists = self.partition(self.seq_len, self.train_level)
        self.logger.info(f"The partition list is: {lists}")
        self.to_visit_frames_dict = {}
        for i in range(self.train_level + 1):
            self.to_visit_frames_dict[i] = lists[i]

        # path_pose_dict = os.path.join(self.result_path, 'pose', 'pose.pth')
        # self.logger.info(f"We save the relative poses to {path_pose_dict}")
        # torch.save(self.pose_dict, path_relative_poses_dict)

        for level_curr in range(self.train_level, -1, -1): 
            to_visit_frames_list = self.to_visit_frames_dict[level_curr]
            num_segments = len(to_visit_frames_list)

            pointer_to_merge_idx = 0
            for segment_idx in range(num_segments): # Each segment corresponds to a 3DGS 
                to_visit_frames = to_visit_frames_list[segment_idx]
                self.logger.info(f"Level {level_curr} || The training frames are: {to_visit_frames}")

                if 'base' in self.pipe_cfg.multi_source_supervision:
                    gs_render = self.gs_render_list[level_curr][segment_idx]
                else:
                    gs_render = self.gs_render_list[segment_idx]
                if hasattr(gs_render, 'global_iteration'):
                    self.global_iteration = gs_render.global_iteration # continue on this global iteration
                else:
                    self.global_iteration = 0
                self.visited_frames = [to_visit_frames[0]]

                if level_curr == self.train_level:
                    # leaf node
                    self.init_leaf_3DGS(to_visit_frames[0], pipe, copy(self.optim_cfg), gs_render=gs_render)
                    gs_render.gaussians.rotate_seq = True
                    gs_render.gaussians.training_setup(self.optim_cfg, fit_pose=True,)
                    gs_render.start_fidx = to_visit_frames[0] # Set the start_fidx
                    gs_render.to_visit_frames = to_visit_frames # Set the start_fidx

                    for fidx in to_visit_frames[1:]:
                        # Update the relative pose to the GS render object
                        rel_pose = self.pose_dict[f'rel_pose_{fidx-1}_to_{fidx}']
                        pose = rel_pose @ gs_render.gaussians.get_RT(fidx-1).detach()
                        gs_render.gaussians.update_RT_seq(pose, fidx)
                        
                        self.visited_frames.append(fidx)

                        # Use the current frame (together with previous frames) to update the global 3DGS
                        self.train_leaf_3DGS(fidx, fidx-1, gs_render=gs_render)

                        gs_render.gaussians.rotate_seq = False

                        # Evaluation
                        psnr_train, render_dict, gt_image = self.render_frame(fidx, gs_render=gs_render)
                        print('Frames {:03d}/{:03d}, PSNR : {:.03f}'.format(fidx, self.seq_len-1, psnr_train))
                        self.logger.info('Frames {:03d}/{:03d}, PSNR : {:.03f}'.format(fidx, self.seq_len-1, psnr_train))
                else:
                    # non-leaf node
                    if 'base' in self.pipe_cfg.multi_source_supervision:
                        self.train_nonleaf_3DGS_phase1(gs_render, self.gs_render_list[level_curr + 1][segment_idx * 2: segment_idx * 2 + 2])
                        for gs_render_child in self.gs_render_list[level_curr + 1][segment_idx * 2: segment_idx * 2 + 2]:
                            del gs_render_child
                        torch.cuda.empty_cache()
            
                    num_iterations_on_indices = self.optim_cfg.num_iterations_per_frame_each_level[level_curr] * len(to_visit_frames)

                    self.train_nonleaf_3DGS_phase2(to_visit_frames, num_iterations=num_iterations_on_indices, gs_render=gs_render)
                
                gs_render.global_iteration = self.global_iteration
                if (segment_idx + 1) % 2 == 0: # even
                    if 'base' in self.pipe_cfg.multi_source_supervision:
                        gs_render_prev = self.gs_render_list[level_curr][segment_idx - 1]
                    else:
                        gs_render_prev = self.gs_render_list[segment_idx - 1]

                    if 'base' in self.pipe_cfg.multi_source_supervision:
                        gs_render_dst = self.gs_render_list[level_curr - 1][(segment_idx - 1) // 2]
                        gs_render_dst.gaussians.restore(gs_render_prev.gaussians.capture(), self.optim_cfg)
                        gs_render_dst.start_fidx = gs_render_prev.start_fidx
                        gs_render_dst.to_visit_frames = gs_render_prev.to_visit_frames # We need to update it later when merging
                    else:
                        gs_render_dst = gs_render_prev
                    pose_between_segments = gs_render_dst.gaussians.get_RT(gs_render.start_fidx)
                    
                    self.merge_two_3DGS(gs_render_dst, gs_render, transform_matrix=pose_between_segments.inverse(), 
                                         to_visit_frames_dst=to_visit_frames_list[segment_idx-1],
                                         to_visit_frames_src=to_visit_frames) # merge the current GS to previous GS
                    # Update the Rt for the merged GS
                    for pose_fidx in to_visit_frames_list[segment_idx]:
                        # Filter out overlapped pose index
                        if pose_fidx in to_visit_frames_list[segment_idx - 1]: 
                            continue 
                        if f'rel_pose_{pose_fidx - 1}_to_{pose_fidx}' not in self.pose_dict:
                            self.compute_relative_pose(pose_fidx, pose_fidx - 1)
                        pose_update = self.pose_dict[f'rel_pose_{pose_fidx - 1}_to_{pose_fidx}'] @ gs_render_dst.gaussians.get_RT(pose_fidx-1).detach()
                        gs_render_dst.gaussians.update_RT_seq(pose_update, pose_fidx)
                    gs_render_dst.global_iteration = 0 # reset global iteration after we merge
                    gs_render_dst.gaussians.training_setup(self.optim_cfg, fix_pos=True,)

                    if 'base' in self.pipe_cfg.multi_source_supervision:
                        gs_render_dst.to_visit_frames = list(sorted(set(gs_render.to_visit_frames + gs_render_dst.to_visit_frames)))
                        self.gs_render_list[level_curr - 1][segment_idx // 2] = gs_render_dst
                    else:
                        del gs_render
                        torch.cuda.empty_cache()
                        self.gs_render_list[pointer_to_merge_idx] = gs_render_dst
                    pointer_to_merge_idx += 1


        if 'base' in self.pipe_cfg.multi_source_supervision:
            self.gs_render = self.gs_render_list[0][0]
        else:
            self.gs_render = self.gs_render_list[0]

        self.evaluate_on_training_images(pipe)
        self.save_checkpoint(save_pose=True, save_global_gs=True)

    def train_nonleaf_3DGS_phase1(self, gs_render, gs_render_children):
        """
            Train the non-leaf 3D Gaussian Splatting (3DGS) model in phase 1.

            Supervision: original training images + virtual view from base 3DGS models

            Args:
                gs_render (GaussianRender): The GaussianRender object used for rendering
                    and training.
                gs_render_children (list of GaussianRender): A list of child GaussianRender
                    objects whose frames are considered during training.
        """

        indices = []
        for child in gs_render_children:
            indices += child.to_visit_frames
        indices = list(sorted(set(indices)))

        optim_cfg = copy(self.optim_cfg)
        optim_cfg.densification_interval = optim_cfg.mss_phase1_densification_interval

        if optim_cfg.mss_phase1_densification_interval is not None:
            optim_cfg.densification_interval = self.optim_cfg.mss_phase1_densification_interval

        num_iterations = optim_cfg.mss_phase1_iteration_per_frame * len(indices)
        if optim_cfg.mss_phase1_densify_until_iter_ratio is not None:
            optim_cfg.densify_until_iter = int(num_iterations * optim_cfg.mss_phase1_densify_until_iter_ratio)

        
        pipe = copy(self.pipe_cfg)
        gs_render.gaussians.rotate_seq = False
        pipe.convert_SHs_python = False

        
        for iteration in range(1, num_iterations+1):
            fidx = random.choice(indices)
            self.global_iteration += 1
            gs_render.gaussians.update_learning_rate(self.global_iteration)
            if random.random() < optim_cfg.mss_phase1_ratio: 
                alpha = random.random()

                if fidx == indices[-1]:
                    fidx -= 1 
                    
                pose_start = gs_render.gaussians.P[fidx].retr()
                pose_end = gs_render.gaussians.P[fidx+1].retr()

                pose_interpolate = self.get_virtual_view(pose_start, pose_end, alpha).cpu()

                # which frame is not important (so I choose 0); we only need the viewpoint to be in the given pose
                viewpoint_cam_pseudo = self.load_viewpoint_cam(0, pose=pose_interpolate, load_depth=True) # wrt the 1st 3DGS

                gs_render_fixed = None
                for gs_render_child in gs_render_children[::-1]: # Due to the overlapping, the overlapped index should appear at late 3DGS, so we iterate from late 3DGS
                    if fidx >= gs_render_child.start_fidx and fidx in gs_render_child.to_visit_frames:
                        gs_render_fixed = gs_render_child
                        break
                if gs_render_fixed is None:
                    raise ValueError
                
                        
                # Get dummy input
                with torch.no_grad():
                    pose_interpolate_wrt_fixed = pose_interpolate.cuda() @ gs_render.gaussians.get_RT(gs_render_fixed.start_fidx).inverse().detach()
                    viewpoint_cam_pseudo_fixed = self.load_viewpoint_cam(0, pose=pose_interpolate_wrt_fixed.cpu(), load_depth=True)
                    render_dict_pseudo = gs_render_fixed.render(viewpoint_cam_pseudo_fixed,
                                                        compute_cov3D_python=self.pipe_cfg.compute_cov3D_python,
                                                        convert_SHs_python=self.pipe_cfg.convert_SHs_python)
                    gt_image_pseudo = render_dict_pseudo['image']

                    
                viewpoint_cam_pseudo.original_image = gt_image_pseudo

                loss, rend_dict_ref, psnr_train = self.train_step(gs_render,
                                                                viewpoint_cam_pseudo,
                                                                self.global_iteration,
                                                                pipe, optim_cfg,
                                                                update_gaussians=True,
                                                                update_cam=False,
                                                                update_distort=self.pipe_cfg.distortion,
                                                                )
            else:
                pose_train = gs_render.gaussians.get_RT(fidx).detach().cpu() if not gs_render.gaussians.rotate_seq else None
                viewpoint_cam = self.load_viewpoint_cam(fidx, pose=pose_train, load_depth=True)

                loss, rend_dict_ref, psnr_train = self.train_step(gs_render,
                                                                viewpoint_cam,
                                                                self.global_iteration,
                                                                pipe, optim_cfg,
                                                                update_gaussians=True,
                                                                update_cam=False,
                                                                update_distort=self.pipe_cfg.distortion,
                                                                )

            if self.global_iteration % 1000 == 0:
                gs_render.gaussians.oneupSHdegree()

            if iteration % 100 == 0:
                loss_display = {k_: v_.item() for k_, v_ in loss.items()}
                self.logger.info(f"[Supervision with base 3DGS] global iter {self.global_iteration} || iter {iteration} || PSNR: {psnr_train} Loss: {loss_display} Number of 3D Gaussians: {gs_render.gaussians.get_xyz.shape[0]}")

    def train_pose_only(self):
        pipe = copy(self.pipe_cfg)
        self.single_step = self.optim_cfg.single_step

        num_iterations = self.single_step * (self.seq_len // 10) * 10
        self.optim_cfg.iterations = num_iterations
        self.optim_cfg.position_lr_max_steps = num_iterations
        self.optim_cfg.opacity_reset_interval = num_iterations // 10
        self.optim_cfg.densify_until_iter = num_iterations
        self.optim_cfg.reset_until_iter = int(num_iterations * 0.8)
        self.optim_cfg.densify_from_iter = self.single_step

        pipe.convert_SHs_python = True

        self.pose_dict = dict()

        os.makedirs(f"{self.result_path}/pose", exist_ok=True)

        self.pipe_cfg.load_pose = None

        self.pose_dict = {}

        self.gs_render.gaussians.init_RT_seq(self.seq_len) # initialize the RT sequence 

        for fidx in range(1, self.seq_len):
            self.compute_relative_pose(fidx, fidx-1)
        
        for fidx in range(1, self.seq_len):
            rel_pose = self.pose_dict[f'rel_pose_{fidx-1}_to_{fidx}']
            pose = rel_pose @ self.gs_render.gaussians.get_RT(fidx-1).detach()
            self.gs_render.gaussians.update_RT_seq(pose, fidx)


        with torch.no_grad():
            self.pose_dict["poses_pred"] = []

            for idx in range(self.seq_len):
                pose = self.gs_render.gaussians.get_RT(idx)
                self.pose_dict["poses_pred"].append(pose.detach().cpu())

        self.pose_dict["poses_pred"] = torch.stack(self.pose_dict["poses_pred"])

        self.logger.info(f"Save pose to {self.result_path}/pose/pose.pth")
        torch.save(self.pose_dict, f"{self.result_path}/pose/pose.pth")

        # path_relative_poses_dict = os.path.join(self.result_path, 'pose', 'relative_poses_dict.pth')
        # torch.save(self.pose_dict, path_relative_poses_dict)

    def eval_nvs(self, ):
        pipe = copy(self.pipe_cfg)
        optim_opt = copy(self.optim_cfg)
        num_epochs = 200
        num_iterations = num_epochs * self.seq_len
        optim_opt.iterations = num_iterations
        optim_opt.position_lr_max_steps = num_iterations
        optim_opt.densify_until_iter = num_iterations // 2
        optim_opt.reset_until_iter = num_iterations // 2
        optim_opt.opacity_reset_interval = num_iterations // 10
        optim_opt.densification_interval = 100
        optim_opt.densify_from_iter = 500

        pipe.convert_SHs_python = True
        optim_opt = copy(self.optim_cfg)
        result_path = os.path.dirname(
            self.model_cfg.model_path).replace('chkpnt', 'test')
        os.makedirs(result_path, exist_ok=True)

        pose_dict = dict()
        pose_dict["poses_gt"] = []
        for seq_data in self.data:
            if self.data_type == "co3d":
                R, t, _, _, _ = self.load_camera(seq_data)
            else:
                try:
                    R = seq_data.R.transpose()
                    t = seq_data.T
                except:
                    R = np.eye(3)
                    t = np.zeros(3)
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            pose_dict["poses_gt"].append(torch.from_numpy(pose))

        max_frame = self.seq_len
        start_frame = 0
        end_frame = max_frame
        if self.model_cfg.model_path != "":
            self.gs_render.gaussians.restore(
                torch.load(self.model_cfg.model_path), self.optim_cfg)
            pose_dict_train = torch.load(
                self.model_cfg.model_path.replace('chkpnt', 'pose').replace('model', 'pose'))
            self.gs_render.gaussians.rotate_seq = True

        sample_rate = 2 if "Family" in result_path else 8
        if 'Family' in result_path:
            pose_test_init = pose_dict_train['poses_pred'][0::sample_rate-1][:max_frame]
        else:
            pose_test_init = pose_dict_train['poses_pred'][int(sample_rate/2)::sample_rate-1][:max_frame]
        self.gs_render.gaussians.init_RT_seq(
            self.seq_len, pose_test_init.float())
        self.gs_render.gaussians.rotate_seq = True
        self.gs_render.gaussians.training_setup(optim_opt,
                                                fix_pos=True,
                                                fix_feat=True,
                                                fit_pose=True,)

        iteration = 0
        # Adjust camera
        for epoch in range(num_epochs):
            for fidx in range(self.seq_len):
                iteration += 1
                self.gs_render.gaussians.rotate_seq = True
                self.gs_render.gaussians.set_seq_idx(fidx)
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=None,
                                                        load_depth=True,
                                                        )
                loss_dict, rend_dict, psnr_train = self.train_step(self.gs_render,
                                                                   viewpoint_cam,
                                                                   iteration, pipe, optim_opt,
                                                                   densify=False,
                                                                   depth_gt=None,
                                                                   update_cam=True,
                                                                   update_gaussians=False,
                                                                   reset=False,
                                                                   )

        psnr_test = 0
        ssim_test = 0
        lpips_test = 0
        psnr_list = []
        ssim_list = []
        lpips_list = []
        with torch.no_grad():
            for fidx in range(self.seq_len):
                self.gs_render.gaussians.rotate_seq = False
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=self.gs_render.gaussians.get_RT(
                                                            fidx).detach().cpu(),
                                                        load_depth=True,
                                                        )

                render_dict = self.gs_render.render(viewpoint_cam,
                                                    compute_cov3D_python=False,
                                                    convert_SHs_python=False)
                gt_image = viewpoint_cam.original_image.cuda()
                psnr_val = psnr(render_dict["image"], gt_image).mean().double()
                ssim_val = ssim(render_dict["image"], gt_image).mean().double()
                lpips_val = lpips(render_dict["image"], gt_image, net_type="vgg").mean().double()
                psnr_test += psnr_val
                ssim_test += ssim_val
                lpips_test += lpips_val
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                lpips_list.append(lpips_val)
                self.visualize(render_dict,
                               f"{result_path}/test/{fidx:04d}.png",
                               gt_image=gt_image, save_ply=False)
        with open(f"{result_path}/test.txt", 'w') as f:
            for i in range(len(psnr_list)):
                f.write(f'{i} {psnr_list[i]:.03f} {ssim_list[i]:.03f} {lpips_list[i]:.03f}\n')
            f.write('PSNR : {:.03f}, SSIM : {:.03f}, LPIPS : {:.03f}'.format(
                    psnr_test / end_frame,
                    ssim_test / end_frame,
                    lpips_test / end_frame))
            f.close()

        print('Number of {:03d} to {:03d} frames: PSNR : {:.03f}, SSIM : {:.03f}, LPIPS : {:.03f}'.format(
            start_frame,
            end_frame,
            psnr_test / end_frame,
            ssim_test / end_frame,
            lpips_test / end_frame))

    def visualize_points(self, gs_render, viewpoint_cam, render_dict=None):
        if render_dict is None:
            render_dict = self.gs_render.render(viewpoint_cam,
                                                    compute_cov3D_python=False,
                                                    convert_SHs_python=False)
        rendered_image = render_dict['image'].clone()
        xyz = gs_render.gaussians.get_xyz
        # distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        in_screen_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # transform points to camera space
        R = torch.tensor(viewpoint_cam.R, device=xyz.device, dtype=torch.float32)
        T = torch.tensor(viewpoint_cam.T, device=xyz.device, dtype=torch.float32)
        xyz_cam = xyz @ R.T + T[None, :] # (N, 3) @ (3, 3) This one is correct! We need to transpose R!
        
        # project to screen space
        valid_depth = xyz_cam[:, 2] > 0.2
        
        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        z = torch.clamp(z, min=0.001)
        
        x = x / z * viewpoint_cam.focal_x + viewpoint_cam.image_width / 2.0
        y = y / z * viewpoint_cam.focal_y + viewpoint_cam.image_height / 2.0

        # Define color and alpha for blending
        overlay_color = torch.tensor([0, 0, 0.75], device=rendered_image.device)  # Blue with intensity
        alpha = 0.5  # Transparency factor        

        visited_pixels = set()
        for i in range(x.shape[0]):
            px, py = int(x[i]), int(y[i])
            if x[i] >= 0 and x[i] < viewpoint_cam.image_width and y[i] >= 0 and y[i] < viewpoint_cam.image_height and (px, py) not in visited_pixels:
                visited_pixels.add((px, py))
                current_color = rendered_image[:, py, px]
                blended_color = alpha * overlay_color + (1 - alpha) * current_color
                rendered_image[:, py, px] = blended_color
        
        rendered_img = Image.fromarray(np.asarray(rendered_image.detach().cpu().permute(1, 2, 0).numpy()* 255.0, dtype=np.uint8)).convert("RGB")
        save_path = os.path.join(self.result_path, f"debug/points_{viewpoint_cam.uid}.png")
        os.makedirs(os.path.join(self.result_path, f"debug"), exist_ok=True)
        rendered_img.save(save_path)

    def eval_pose(self, ):
        pipe = copy(self.pipe_cfg)
        optim_opt = copy(self.optim_cfg)
        result_path = os.path.dirname(self.model_cfg.model_path).replace('chkpnt', 'pose')
        os.makedirs(result_path, exist_ok=True)
        pose_path = os.path.join(result_path, 'pose.pth')
        poses = torch.load(pose_path)
        poses_pred = poses['poses_pred'].inverse().cpu()
        
        # pose_path = os.path.join(result_path, 'relative_poses_dict.pth')
        # relative_poses = torch.load(pose_path)
        # poses = [torch.eye(4, 4)]
        # for i in range(len(relative_poses)):
        #     poses.append(relative_poses[f'rel_pose_{i}_to_{i+1}'].cpu() @ poses[-1])
        # poses = torch.stack(poses, dim=0)
        # poses_pred = poses['poses_pred'].inverse().cpu()
        
        # Get ground-truth pose from the dataset
        poses_gt = []
        for seq_data in self.train_cam_infos:
            if self.data_type == "co3d":
                R, t, _, _, _ = self.load_camera(seq_data)
            else:
                try:
                    R = seq_data.R.transpose()
                    t = seq_data.T
                except:
                    R = np.eye(3)
                    t = np.zeros(3)
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            poses_gt.append(torch.from_numpy(pose))
        poses_gt = torch.stack(poses_gt)
        
        # poses_gt_c2w = poses['poses_gt'].inverse().cpu()
        poses_gt_c2w = poses_gt.inverse().cpu()
        poses_gt = poses_gt_c2w[:len(poses_pred)].clone()
        # align scale first (we do this because scale differennt a lot)
        trans_gt_align, trans_est_align, _ = self.align_pose(poses_gt[:, :3, -1].numpy(),
                                                             poses_pred[:, :3, -1].numpy())
        poses_gt[:, :3, -1] = torch.from_numpy(trans_gt_align)
        poses_pred[:, :3, -1] = torch.from_numpy(trans_est_align)

        c2ws_est_aligned = align_ate_c2b_use_a2b(poses_pred, poses_gt)
        ate = compute_ATE(poses_gt.cpu().numpy(),
                          c2ws_est_aligned.cpu().numpy())
        rpe_trans, rpe_rot = compute_rpe(
            poses_gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
        print("{0:.3f}".format(rpe_trans*100),
              '&' "{0:.3f}".format(rpe_rot * 180 / np.pi),
              '&', "{0:.3f}".format(ate))
        plot_pose(poses_gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy(), pose_path)
        with open(f"{result_path}/pose_eval.txt", 'w') as f:
            f.write("RPE_trans: {:.03f}, RPE_rot: {:.03f}, ATE: {:.03f}".format(
                rpe_trans*100,
                rpe_rot * 180 / np.pi,
                ate))
            f.close()

    def align_pose(self, pose1, pose2):
        mtx1 = np.array(pose1, dtype=np.double, copy=True)
        mtx2 = np.array(pose2, dtype=np.double, copy=True)

        if mtx1.ndim != 2 or mtx2.ndim != 2:
            raise ValueError("Input matrices must be two-dimensional")
        if mtx1.shape != mtx2.shape:
            raise ValueError("Input matrices must be of same shape")
        if mtx1.size == 0:
            raise ValueError("Input matrices must be >0 rows and >0 cols")

        # translate all the data to the origin
        mtx1 -= np.mean(mtx1, 0)
        mtx2 -= np.mean(mtx2, 0)

        norm1 = np.linalg.norm(mtx1)
        norm2 = np.linalg.norm(mtx2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Input matrices must contain >1 unique points")

        # change scaling of data (in rows) such that trace(mtx*mtx') = 1
        mtx1 /= norm1
        mtx2 /= norm2

        # transform mtx2 to minimize disparity
        R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
        mtx2 = mtx2 * s

        return mtx1, mtx2, R

    def render_nvs(self, traj_opt='bspline', N_novel_imgs=120, degree=100):
        result_path = os.path.dirname(
            self.model_cfg.model_path).replace('chkpnt', 'nvs')
        os.makedirs(result_path, exist_ok=True)
        self.gs_render.gaussians.restore(torch.load(self.model_cfg.model_path), self.optim_cfg)
        pose_dict_train = torch.load(
            self.model_cfg.model_path.replace('chkpnt', 'pose'))
        poses_pred_w2c_train = pose_dict_train['poses_pred'].cpu()
        if traj_opt == 'bspline':
            i_train = self.i_train
            if "co3d" in self.model_cfg.source_path:
                poses_pred_w2c_train = poses_pred_w2c_train[:100]
                i_train = self.i_train[:100]
            c2ws = interp_poses_bspline(poses_pred_w2c_train.inverse(), N_novel_imgs,
                                        i_train, degree)
            w2cs = c2ws.inverse()

        self.gs_render.gaussians.rotate_seq = False
        render_dir = f"{result_path}/{traj_opt}"
        os.makedirs(render_dir, exist_ok=True)
        for fidx, pose in enumerate(w2cs):
            viewpoint_cam = self.load_viewpoint_cam(10,
                                                    pose=pose,
                                                    )
            render_dict = self.gs_render.render(viewpoint_cam,
                                                compute_cov3D_python=False,
                                                convert_SHs_python=False)
            self.visualize(render_dict,
                           f"{render_dir}/img_out/{fidx:04d}.png",
                           save_ply=False)

        imgs = []
        for img in sorted(glob.glob(os.path.join(render_dir, "img_out", "*.png"))):
            if "depth" in img:
                continue
            rgb = cv2.imread(img)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(img.replace(".png", "_depth.png"))
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
            rgb = np.hstack([rgb, depth])
            imgs.append(rgb)

        imgs = np.stack(imgs, axis=0)

        video_out_dir = os.path.join(render_dir, 'video_out')
        if not os.path.exists(video_out_dir):
            os.makedirs(video_out_dir)
        imageio.mimwrite(os.path.join(
            video_out_dir, f'{self.category}_{self.seq_name}_ours.mp4'), imgs, fps=30, quality=9)

    def save_model(self, epoch):
        pass

    def compute_loss(self,
                     render_dict,
                     viewpoint_cam,
                     pipe_opt,
                     iteration,
                     use_reproject=False,
                     use_matcher=False,
                     ref_fidx=None,
                     **kwargs):
        loss = 0.0
        if "image" in render_dict:
            image = render_dict["image"]
            gt_image = viewpoint_cam.original_image.cuda()
        if "depth" in render_dict:
            depth = render_dict["depth"]
            depth[depth < self.near] = self.near
            fidx = viewpoint_cam.uid
            kwargs['depth_pred'] = depth

        loss_dict = self.loss_func(image, gt_image, **kwargs)
        return loss_dict

    def visualize(self, render_pkg, filename, gt_image=None, gt_depth=None, save_ply=False, save_depth=False):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if "depth" in render_pkg and save_depth:
            rend_depth = Image.fromarray(
                colorize(render_pkg["depth"].detach().cpu().numpy(),
                         cmap='magma_r')).convert("RGB")
            if gt_depth is not None:
                gt_depth = Image.fromarray(
                    colorize(gt_depth.detach().cpu().numpy(),
                             cmap='magma_r')).convert("RGB")
                rend_depth = Image.fromarray(np.hstack([np.asarray(gt_depth),
                                                        np.asarray(rend_depth)]))
            rend_depth.save(filename.replace(".png", "_depth.png"))
        if "acc" in render_pkg:
            rend_acc = Image.fromarray(
                colorize(render_pkg["acc"].detach().cpu().numpy(),
                         cmap='magma_r')).convert("RGB")
            rend_acc.save(filename.replace(".png", "_acc.png"))

        rend_img = Image.fromarray(
            np.asarray(render_pkg["image"].detach().cpu().permute(1, 2, 0).numpy()
                       * 255.0, dtype=np.uint8)).convert("RGB")
        if gt_image is not None:
            gt_image = Image.fromarray(
                np.asarray(
                    gt_image.permute(1, 2, 0).cpu().numpy() * 255.0,
                    dtype=np.uint8)).convert("RGB")
            rend_img = Image.fromarray(np.hstack([np.asarray(gt_image),
                                                  np.asarray(rend_img)]))
        rend_img.save(filename)

        if save_ply:
            points = self.gs_render.gaussians._xyz.detach().cpu().numpy()
            pcd_data = o3d.geometry.PointCloud()
            pcd_data.points = o3d.utility.Vector3dVector(points)
            pcd_data.colors = o3d.utility.Vector3dVector(np.ones_like(points))
            o3d.io.write_point_cloud(
                filename.replace('.png', '.ply'), pcd_data)
    
    def partition(self, n, level, overlap=2):
        """
            NOTE: We emprically found that the partition strategy does not have a significant effect on the results.
                  Therefore, evenly sampling is recommended.
                  The original paper uses v1 version, but it achieves similar peformance.
            n: total number of frames
            level: the level of partition
        """
        if self.pipe_cfg.partition_strategy == 'v1':
            relative_diff = []
            for idx in range(n - 1):
                relative_pose = self.pose_dict[f'rel_pose_{idx}_to_{idx+1}']
                diff = self._calculate_relative_pose_size(relative_pose)
                relative_diff.append((diff, idx))
            num_segment = 2 ** level
            len_segment = n // num_segment
            num_subsegment = num_segment * 4
            len_subsegment = n // num_subsegment # Each subsegment corresponds to frames of length "len_subsegment"

            key_indices = []
            for i in range(num_segment - 1):
                idx = (i+1) * len_segment
                partition_idx = sorted(relative_diff[idx-len_subsegment:idx+len_subsegment+1])[-1][1]
                key_indices.append(partition_idx)

            result = dict()
            for i in range(level, -1, -1):
                result[i] = []
                if i == level:
                    prev = 0
                    for key_idx in key_indices:
                        result[i].append(list(range(prev, key_idx+1+overlap)))
                        prev = key_idx+1
                    result[i].append(list(range(prev, n)))
                else:
                    for idx in range(0, len(result[i+1]), 2):
                        l1, l2 = result[i+1][idx], result[i+1][idx+1]
                        result[i].append(sorted(list(set(l1 + l2))))
            assert result[0][0] == list(range(n))
            return result
        else:
            assert level <= 3, "level must be <= 3. But it should be easy to transfer to higher level"
            result = dict()
            indices = list(range(n))
            result[0] = [indices]
            if level == 0: return result
            result[1] = [indices[:n//2+1], indices[n//2-1:]]
            if level == 1: return result
            result[2] = []
            for ind in result[1]:
                result[2].append(ind[:len(ind)//2+1])
                result[2].append(ind[len(ind)//2-1:])
            if level == 2: return result
            result[3] = []
            for ind in result[2]:
                result[3].append(ind[:len(ind)//2+1])
                result[3].append(ind[len(ind)//2-1:])
            return result


    # Function to calculate the relative translation and rotation between two camera poses
    def _calculate_relative_pose_size(self, pose, translation_weight=1.0, rotation_weight=1.0):
        """
            NOTE: Since the way we partition the video does not significantly affect the final performance, 
                  the method used to calculate the relative pose is not crucial. 
                  Therefore, the code provided below is not essential.

            pose1 and pose2 are 4x4 transformation matrices representing the camera poses.
            This function returns a single combined value that accounts for both translation and rotation.
            
            translation_weight and rotation_weight are used to balance the contribution of translation and rotation
            to the overall pose difference.
        """
        t = pose[:3, 3]
        
        translation_magnitude = torch.norm(t)
        
        # Extract rotation matrices (top-left 3x3 part of the transformation matrix)
        R = pose[:3, :3]
        
        # Calculate the angle of the relative rotation using trace
        trace = torch.trace(R)
        rotation_angle = torch.acos((trace - 1) / 2)
        
        # Combine the translation magnitude and rotation angle into a single value
        combined_pose_difference = (translation_weight * translation_magnitude) + (rotation_weight * rotation_angle)
        
        return combined_pose_difference, translation_magnitude, rotation_angle
        
    @staticmethod
    def calc_importance(gs_render, cameras, pipe):
        """
            Reference: https://github.com/KeKsBoTer/c3dgs/blob/master/compress.py
        """
        gaussians = gs_render.gaussians

        cov3d = gaussians.covariance_activation(gaussians.get_scaling.detach(), 1.0, gaussians._rotation.detach()).requires_grad_(True)

        h1 = gaussians._features_dc.register_hook(lambda grad: grad.abs())
        h2 = gaussians._features_rest.register_hook(lambda grad: grad.abs())
        h3 = cov3d.register_hook(lambda grad: grad.abs())

        gaussians._features_dc.grad = None
        gaussians._features_rest.grad = None
        num_pixels = 0
        for viewport_cam in cameras:
            rendering = gs_render.render(
                    viewport_cam,
                    compute_cov3D_python=pipe.compute_cov3D_python,
                    convert_SHs_python=pipe.convert_SHs_python,
                    override_color=None)['image']

            loss = rendering.sum()
            loss.backward()
            num_pixels += rendering.shape[1]*rendering.shape[2]

        importance = torch.cat(
            [gaussians._features_dc.grad, gaussians._features_rest.grad],
            1,
        ).flatten(-2)/num_pixels
        h1.remove()
        h2.remove()
        h3.remove()
        torch.cuda.empty_cache()
        return importance.detach() 


    def evaluate_on_training_images(self, pipe):
        os.makedirs(f"{self.result_path}/eval", exist_ok=True)
        with torch.no_grad():
            psnr_test = 0.0
            self.pose_dict["poses_pred"] = []
            self.render_depth = OrderedDict()
            self.gs_render.gaussians.rotate_seq = False
            self.gs_render.gaussians.rotate_xyz = False

            for val_idx in range(self.seq_len):
                viewpoint_cam = self.load_viewpoint_cam(val_idx,
                                                        pose=self.gs_render.gaussians.get_RT(
                                                            val_idx).detach().cpu(),
                                                        )
                render_dict = self.gs_render.render(viewpoint_cam,
                                                    compute_cov3D_python=pipe.compute_cov3D_python,
                                                    convert_SHs_python=pipe.convert_SHs_python)
                self.render_depth[val_idx] = render_dict["depth"]
                gt_image = viewpoint_cam.original_image.cuda()
                psnr_curr = psnr(render_dict["image"],
                                    gt_image).mean().double()
                psnr_test += psnr_curr
                self.visualize(render_dict,
                                f"{self.result_path}/eval/{val_idx:03d}.png",
                                gt_image=gt_image, save_ply=False)
                self.logger.info(f"Frame {val_idx}: PSNR = {psnr_curr}")
            print('Number of {:03d} to {:03d} frames: PSNR : {:.03f}'.format(
                0,
                self.seq_len,
                psnr_test / (self.seq_len)))

    def save_checkpoint(self, save_pose=True, save_global_gs=True):
        for idx in range(self.seq_len):
            pose = self.gs_render.gaussians.get_RT(idx)
            self.pose_dict["poses_pred"].append(pose.detach().cpu())

        self.pose_dict["poses_pred"] = torch.stack(self.pose_dict["poses_pred"])
        if save_pose:
            os.makedirs(f"{self.result_path}/pose", exist_ok=True)
            self.logger.info(f"Save pose to {self.result_path}/pose/pose.pth")
            torch.save(self.pose_dict, f"{self.result_path}/pose/pose.pth")
        if save_global_gs:
            os.makedirs(f"{self.result_path}/chkpnt", exist_ok=True)
            self.logger.info(f"Save 3DGS to {self.result_path}/chkpnt/model.pth")
            torch.save(self.gs_render.gaussians.capture(), f"{self.result_path}/chkpnt/model.pth")