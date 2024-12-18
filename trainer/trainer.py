# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import pdb
from kornia.geometry.depth import depth_to_3d, depth_to_normals
from pytorch3d.utils import opencv_from_cameras_projection
from pytorch3d.renderer import PerspectiveCameras
import pytorch3d
from utils.image_utils import psnr, colorize
from utils.loss_utils import l1_loss, ssim
from scene.cameras import Camera
from utils.graphics_utils import BasicPointCloud, focal2fov, procrustes, fov2focal
from scene.gaussian_model import GaussianModel
from scene import Scene
from gaussian_renderer import render
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.dataset_readers import sceneLoadTypeCallbacks, CameraInfo, read_intrinsics_binary
import glob
from copy import copy
import open3d as o3d
from einops import rearrange
from PIL import Image
import os
from tqdm import tqdm
import math
import numpy as np
import cv2
import imageio
from collections import defaultdict, OrderedDict
import json
import gzip
import logging
import torch
import torch.nn.functional as F
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)


class GaussianTrainer(object):

    def __init__(self, data_root, model_cfg, pipe_cfg, optim_cfg, logger_name=None, modelhidden_cfg=None):
        self.model_cfg = model_cfg
        self.pipe_cfg = pipe_cfg
        self.optim_cfg = optim_cfg
        self.modelhidden_cfg = modelhidden_cfg
        self.seq_name = model_cfg.seq_name
        self.category = model_cfg.category
        self.data_root = data_root.split(self.category)[0]
        self.data_type = model_cfg.data_type
        self.depth_model_type = model_cfg.depth_model_type
        self.rgb_images = OrderedDict()
        self.render_depth = OrderedDict()
        self.render_image = OrderedDict()
        self.mono_depth = OrderedDict()
            
        if self.pipe_cfg.train_pose_mode == 'vfi' or 'vfi' in self.pipe_cfg.multi_source_supervision:
            self.vfi = OrderedDict()

            from scene.vfi_model import Model

            model = Model()
            model.load_state_dict(torch.load('pretrained/vfi/IFRNet_Vimeo90K.pth'))
            model.eval()
            model.cuda()
            self.vfi_model = model
            self.vfi = OrderedDict()

        expname = self.model_cfg.expname
        if 'eval' in self.model_cfg.mode:
            s = os.path.dirname(self.model_cfg.model_path)
            self.result_path = s.rstrip('/').replace(s.rstrip('/').split('/')[-1], '')
        else:
            self.result_path = f"output/{expname}/{self.category}_{self.seq_name}"
        self.setup_logger(logger_name)

        self.logger.info(f"model cfg: {self.model_cfg.get_all_members_as_string()}")
        self.logger.info(f"pipe cfg: {self.pipe_cfg.get_all_members_as_string()}")
        self.logger.info(f"optim cfg: {self.optim_cfg.get_all_members_as_string()}")

        self.setup_dataset()
        self.setup_depth_predictor()


    def setup_logger(self, logger_name=None):
        self.logger = logging.getLogger('my_logger')
        self.logger.setLevel(logging.DEBUG)
        os.makedirs('output', exist_ok=True)
        print('result_path:', self.result_path)
        os.makedirs(self.result_path, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(self.result_path, 'output.log'))
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)

    def load_camera(self, data, scale=1.0):
        """
        Load a camera from a CO3D annotation.
        """

        principal_point = torch.tensor(
            data["viewpoint"]["principal_point"], dtype=torch.float)
        focal_length = torch.tensor(
            data["viewpoint"]["focal_length"], dtype=torch.float)
        half_image_size_wh_orig = (
            torch.tensor(
                list(reversed(data["image"]["size"])), dtype=torch.float) / 2.0
        )
        format_ = data["viewpoint"]["intrinsics_format"]
        if format_.lower() == "ndc_norm_image_bounds":
            # this is e.g. currently used in CO3D for storing intrinsics
            rescale = half_image_size_wh_orig
        elif format_.lower() == "ndc_isotropic":
            rescale = half_image_size_wh_orig.min()
        else:
            raise ValueError(f"Unknown intrinsics format: {format}")

        principal_point_px = half_image_size_wh_orig - principal_point * rescale
        focal_length_px = focal_length * rescale

        # now, convert from pixels to PyTorch3D v0.5+ NDC convention
        out_size = list(reversed(data["image"]["size"]))

        half_image_size_output = torch.tensor(
            out_size, dtype=torch.float) / 2.0
        half_min_image_size_output = half_image_size_output.min()

        # rescaled principal point and focal length in ndc
        principal_point = (
            half_image_size_output - principal_point_px * scale
        ) / half_min_image_size_output
        focal_length = focal_length_px * scale / half_min_image_size_output

        camera = PerspectiveCameras(
            focal_length=focal_length[None],
            principal_point=principal_point[None],
            R=torch.tensor(data["viewpoint"]["R"], dtype=torch.float)[None],
            T=torch.tensor(data["viewpoint"]["T"], dtype=torch.float)[None],
        )

        img_size = torch.tensor(data["image"]["size"], dtype=torch.float)[None]
        R, t, intr_mat = opencv_from_cameras_projection(camera, img_size)
        FoVy = focal2fov(intr_mat[0, 1, 1], img_size[0, 0])
        FoVx = focal2fov(intr_mat[0, 0, 0], img_size[0, 1])

        return R[0].numpy(), t[0].numpy(), FoVx, FoVy, intr_mat[0].numpy()

    def setup_depth_predictor(self,):
        # we recommand to use the following depth models:
        # - "midas" for the Tank and Temples dataset
        # - "zoe" for the CO3D dataset
        # - "depth_anything" for the custom dataset
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.depth_model_type == "zoe":
            repo = "isl-org/ZoeDepth"
            model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
            zoe = model_zoe_n.to(device)
            self.depth_model = zoe
        elif self.depth_model_type == "depth_anything":
            from torchvision.transforms import Compose
            from submodules.DepthAnything.depth_anything.dpt import DepthAnything
            from submodules.DepthAnything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
            encoder = 'vits' # can also be 'vitb' or 'vitl'
            depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()
            # depth_anything = DepthAnything.from_pretrained('checkpoints/depth_anything_metric_depth_outdoor', local_files_only=True).eval()

            self.depth_transforms = Compose([
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
            self.depth_model = depth_anything
        else:
            midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            midas.to(device)
            midas.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.depth_transforms = midas_transforms.dpt_transform
            self.depth_model = midas
        self.mono_depth = OrderedDict()

    def predict_depth(self, img):
        if self.depth_model_type == "zoe":
            self.logger.info(f"We use zoe to predict the depth")
            depth = self.depth_model.infer_pil(Image.fromarray(img.astype(np.uint8)),
                                               output_type='tensor')
        elif self.depth_model_type == "depth_anything":
            image = self.depth_transforms({'image': img/255.})['image']
            image = torch.from_numpy(image).unsqueeze(0)
            # depth shape: 1xHxW
            prediction = self.depth_model(image)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().detach()
            
            scale = 0.0305
            shift = 0.15
            depth = scale * prediction + shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_batch = self.depth_transforms(img).to(device)
            with torch.no_grad():
                prediction = self.depth_model(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            scale = 0.000305
            shift = 0.1378
            depth = scale * prediction + shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth

        return depth


    def predict_vfi(self, image1, image2):
        """
            Output: 3 H W
        """
        from utils.vfi_utils import InputPadder

        assert image1.max() <= 1.1 # Simple check
        if len(image1.shape) == 3:
            image1 = image1[None]
            image2 = image2[None]

        padder = InputPadder(image1.shape, 16)
        image1_padded, image2_padded = padder.pad(image1, image2)

        embt = torch.tensor(1/2).float().view(1, 1, 1, 1).cuda()
        with torch.no_grad():
            pred = self.vfi_model.inference(image1_padded, image2_padded, embt).squeeze().detach()
            pred = padder.unpad(pred)

        return pred



    def setup_dataset(self):
        if self.model_cfg.data_type == "co3d":
            sequences = defaultdict(list)
            print(f"self.data_root: {self.data_root}; self.category: {self.category}")
            dataset = json.loads(gzip.GzipFile(os.path.join(self.data_root, self.category, 
                                                            self.seq_name.split('_')[0],  # teddybear_34_1403_4393 --> teddybear/34_1403_4393
                                                            "frame_annotations.jgz"), "rb").read().decode("utf8"))
            for data in dataset:
                sequences[data["sequence_name"]].append(data)
            self.subseq_name = '_'.join(self.seq_name.split('_')[1:]) # seq_name: teddybear_34_1403_4393; subseq_name: 34_1403_4393
            seq_data = sequences[self.subseq_name]
            if self.model_cfg.eval:
                sample_rate = 8
                ids = np.arange(len(seq_data))
                self.i_test = ids[int(sample_rate/2)::sample_rate]
                self.i_train = np.array(
                    [i for i in ids if i not in self.i_test])
                self.train_cam_infos = [c for idx, c in enumerate(
                    seq_data) if idx in self.i_train]
                self.test_cam_infos = [c for idx, c in enumerate(
                    seq_data) if idx in self.i_test]
                train_image_names = [c["image"]["path"]
                                     for c in self.train_cam_infos]
                test_image_names = [c["image"]["path"] for c in self.test_cam_infos]
                print("Train images: ", len(train_image_names))
                print(train_image_names)
                print("Test images: ", len(test_image_names))
                print(test_image_names)
                if "eval" in self.model_cfg.mode:
                    self.data = self.test_cam_infos
                else:
                    self.data = self.train_cam_infos
            else:
                self.data = seq_data
            self.seq_len = len(self.data)
        elif self.model_cfg.data_type == "custom":
            assert False, "Not implemented"
            source_path = self.model_cfg.source_path
            cameras_intrinsic_file = os.path.join(source_path, "sparse/0", "cameras.bin")
            max_frames = 300
            images = sorted(glob.glob(os.path.join(source_path, "images/*.jpg")))
            if len(images) > max_frames:
                interval = len(images) // max_frames
                images = images[::interval]
            print("Total images: ", len(images))
            width, height = Image.open(images[0]).size
            if os.path.exists(cameras_intrinsic_file):
                cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
                intr = cam_intrinsics[1]
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                height = intr.height
                width = intr.width
                intr_mat = np.array(
                    [[focal_length_x, 0, width/2], [0, focal_length_y, height/2], [0, 0, 1]])
            else:
                # use some hardcoded values
                fov = 79.0
                FoVx = fov * math.pi / 180
                intr_mat = np.eye(3)
                intr_mat[0, 0] = fov2focal(FoVx, width)
                intr_mat[1, 1] = fov2focal(FoVx, width)
                intr_mat[0, 2] = width / 2
                intr_mat[1, 2] = height / 2

            if min(width, height) > 1000:
                width = width // 2
                height = height // 2
            
            intr_mat[:2, :] /= 2
            self.intrinsic = intr_mat


            sample_rate = 8
            ids = np.arange(len(images))
            self.i_test = ids[int(sample_rate/2)::sample_rate]
            self.i_train = np.array([i for i in ids if i not in self.i_test])
            if "eval" in self.model_cfg.mode:
                self.data = [images[i] for i in self.i_test]
            else:
                self.data = [images[i] for i in self.i_train]
            self.seq_len = len(self.data)
        elif self.model_cfg.data_type == "blender":
            assert False, "Not implemented"
            # self.logger.debug(f'{__file__}: we are here ')
            source_path = self.model_cfg.source_path # data/dnerf/bouncingballs
            model_cfg = copy(self.model_cfg)
            model_cfg.model_path = ""
            gaussians = GaussianModel(3)
            scene = Scene(model_cfg, gaussians, shuffle=False)
            test_views = scene.getTestCameras().copy()
            train_views = scene.getTrainCameras().copy()
            all_views = train_views + test_views

            ids = np.arange(len(all_views))
            self.i_test = ids[len(train_views):]
            self.i_train = ids[:len(train_views)]
            if "eval" in self.model_cfg.mode:
                viewpoint_stack = test_views
                print("Test images: ", len(viewpoint_stack))
                image_name = [c.image_name for c in viewpoint_stack]
                print(image_name)
            else:
                viewpoint_stack = train_views
                print("Train images: ", len(viewpoint_stack))
                image_name = [c.image_name for c in viewpoint_stack]
                print(image_name)
            self.data = viewpoint_stack 
            self.seq_len = len(viewpoint_stack)
        elif self.model_cfg.data_type == "images_only":
            source_path = self.model_cfg.source_path # data/dnerf/bouncingballs
            model_cfg = copy(self.model_cfg)
            model_cfg.model_path = ""
            gaussians = GaussianModel(3)
            scene = Scene(model_cfg, gaussians, shuffle=False)
            test_views = scene.getTestCameras().copy()
            train_views = scene.getTrainCameras().copy()
            all_views = train_views + test_views

            ids = np.arange(len(all_views))
            self.i_test = ids[len(train_views):]
            self.i_train = ids[:len(train_views)]
            print(f"self.i_test: {self.i_test}")
            print(f"self.i_train: {self.i_train}")
            if "eval" in self.model_cfg.mode:
                viewpoint_stack = test_views
                print("Test images: ", len(viewpoint_stack))
                image_name = [c.image_name for c in viewpoint_stack]
                print(image_name)
            else:
                viewpoint_stack = train_views
                print("Train images: ", len(viewpoint_stack))
                image_name = [c.image_name for c in viewpoint_stack]
                print(image_name)
            self.data = viewpoint_stack # self.data is probably a list of cameras
            self.train_cam_infos = train_views
            self.test_cam_infos = test_views
            self.seq_len = len(viewpoint_stack)
        else:
            source_path = self.model_cfg.source_path
            model_cfg = copy(self.model_cfg)
            model_cfg.model_path = ""
            gaussians = GaussianModel(3)
            scene = Scene(model_cfg, gaussians, shuffle=False)
            test_views = scene.getTestCameras().copy()
            train_views = scene.getTrainCameras().copy()
            all_views = train_views + test_views

            # sample_rate = 8
            sample_rate = 2 if "Family" in source_path else 8
            ids = np.arange(len(all_views))
            self.i_test = ids[int(sample_rate/2)::sample_rate]
            self.i_train = np.array([i for i in ids if i not in self.i_test])
            if "eval" in self.model_cfg.mode:
                viewpoint_stack = test_views
                print("Test images: ", len(viewpoint_stack))
                image_name = [c.image_name for c in viewpoint_stack]
                print(image_name)
            else:
                viewpoint_stack = train_views
                print("Train images: ", len(viewpoint_stack))
                image_name = [c.image_name for c in viewpoint_stack]
                print(image_name)
            self.data = viewpoint_stack
            self.train_cam_infos = train_views
            self.test_cam_infos = test_views
            self.seq_len = len(viewpoint_stack)

    def setup_model(self, pcd):
        radius = np.linalg.norm(pcd.points, axis=1).max()
        gaussians = GaussianModel(sh_degree=3)
        gaussians.create_from_pcd(pcd, math.ceil(radius))
        self.model = gaussians
        self.radius = radius

    def setup_optimizer(self, optim_cfg):
        pass

    def prepare_data_co3d(self, idx, down_sample=True,
                          orthogonal=True, learn_pose=False,
                          pose=None, load_depth=True,
                          load_gt=False, load_vfi=False):
        cam_info = dict()
        image_name = self.data[idx]["image"]["path"]
        image_path = os.path.join(self.data_root, 'co3d', image_name)
        mask_name = self.data[idx]["mask"]["path"]
        mask_path = os.path.join(self.data_root, 'co3d', mask_name)
        mask = torch.from_numpy(np.asarray(Image.open(mask_path))
                                / 255.0).unsqueeze(0).repeat(3, 1, 1).float()
        mask = mask > 0.5
        if not self.use_mask:
            mask = None

        R, t, FoVx, FoVy, intr_mat = self.load_camera(self.data[idx])
        pose_src = np.eye(4)
        pose_src[:3, :3] = R
        pose_src[:3, 3] = t
        cam_info["gt_pose"] = copy(pose_src)
        cam_info["intrinsics"] = intr_mat
        cam_info["FoVx"] = FoVx
        cam_info["FoVy"] = FoVy
        cam_info["R"] = R
        cam_info["t"] = t

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        if orthogonal:
            R = np.eye(3)
            t = np.zeros(3)

        elif learn_pose:
            pose_learn = self.camera_model(idx)
            R = pose_learn[:3, :3].detach().cpu().numpy()
            t = pose_learn[:3, 3].detach().cpu().numpy()

        if pose is not None:
            R = pose[:3, :3].numpy()
            t = pose[:3, 3].numpy()

        cam_info["R"] = R
        cam_info["t"] = t
        if load_depth:
            if idx not in self.mono_depth:
                depth_tensor = self.predict_depth(np.asarray(image))
                self.mono_depth[idx] = depth_tensor.cuda()
            else:
                depth_tensor = self.mono_depth[idx]
        else:
            w, h = image.size
            depth_tensor = torch.ones((h, w))
            self.mono_depth[idx] = depth_tensor.cuda()

        intr_mat_tensor = torch.from_numpy(
            intr_mat).float().to(depth_tensor.device)
        pts = depth_to_3d(depth_tensor[None, None],
                          intr_mat_tensor[None],
                          normalize_points=False)

        points = pts[0].permute(1, 2, 0).cpu().reshape(-1, 3)
        if not orthogonal:
            pose_inv = torch.inverse(pose)
            points = (pose_inv[:3, :3] @ points.t()).t() + pose_inv[:3, 3]
        points = points.numpy()

        colors = np.asarray(image).reshape(-1, 3) / 255.0
        colors_torch = torch.from_numpy(np.asarray(
            image) / 255.0).permute(2, 0, 1).float()
        
        vfi = None
        if load_vfi:
            if idx + 1 < len(self.data):
                if f"{idx}_to_{idx+1}" not in self.vfi:
                    image_path1 = os.path.join(self.data_root, 'co3d', self.data[idx]['image']['path'])
                    image_path2 = os.path.join(self.data_root, 'co3d', self.data[idx + 1]['image']['path'])
                    image1 = imageio.imread(image_path1)
                    image2 = imageio.imread(image_path2)
                    image1 = (torch.tensor(image1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
                    image2 = (torch.tensor(image2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()

                    vfi = self.predict_vfi(image1, image2)

                    self.vfi[f"{idx}_to_{idx+1}"] = vfi.cuda() # 3, h, w

                else:
                    vfi = self.vfi[f"{idx}_to_{idx+1}"]
            else:
                w, h = image.size
                vfi = torch.ones((3, h, w))
                vfi = vfi.cuda()
                self.vfi[idx] = vfi


            depth_tensor_vfi = self.predict_depth(np.asarray(vfi.squeeze().detach().cpu().permute(1, 2, 0).numpy()* 255.0, dtype=np.uint8))

            pts_vfi = depth_to_3d(depth_tensor_vfi[None, None],
                            intr_mat_tensor[None].to(depth_tensor_vfi.device),
                            normalize_points=False)


            points_vfi = pts_vfi[0].permute(1, 2, 0).cpu().reshape(-1, 3)
            if not orthogonal:
                assert False
                pose_inv = torch.inverse(pose)
                points = (pose_inv[:3, :3] @ points.t()).t() + pose_inv[:3, 3]
            points_vfi = points_vfi.numpy()

            colors_vfi = vfi.squeeze().detach().cpu().permute(1, 2, 0).numpy().reshape(-1, 3)
            viewpoint_camera_vfi = Camera(idx + 0.5, R, t, FoVx, FoVy, vfi.squeeze(),
                                    gt_alpha_mask=None, image_name=None,
                                    intrinsics=intr_mat,
                                    uid=idx+0.5, is_co3d=True)

            pcd_data_vfi = o3d.geometry.PointCloud()
            pcd_data_vfi.points = o3d.utility.Vector3dVector(points_vfi)
            pcd_data_vfi.colors = o3d.utility.Vector3dVector(colors_vfi)
            pcd_data_vfi.estimate_normals()
            if down_sample:
                pcd_data_vfi = pcd_data_vfi.voxel_down_sample(voxel_size=0.01)
            colors_vfi = np.asarray(pcd_data_vfi.colors, dtype=np.float32)
            points_vfi = np.asarray(pcd_data_vfi.points, dtype=np.float32)
            normals_vfi = np.asarray(pcd_data_vfi.normals, dtype=np.float32)
            pcd_vfi = BasicPointCloud(points_vfi, colors_vfi, normals_vfi)

        
        viewpoint_camera = Camera(idx, R, t, FoVx, FoVy, colors_torch,
                                  gt_alpha_mask=mask, image_name=image_name,
                                  intrinsics=intr_mat,
                                  uid=idx, is_co3d=True)
        pcd_data = o3d.geometry.PointCloud()
        pcd_data.points = o3d.utility.Vector3dVector(points)
        pcd_data.colors = o3d.utility.Vector3dVector(colors)
        pcd_data.estimate_normals()
        if down_sample:
            pcd_data = pcd_data.voxel_down_sample(voxel_size=0.01)
        colors = np.asarray(pcd_data.colors, dtype=np.float32)
        points = np.asarray(pcd_data.points, dtype=np.float32)
        normals = np.asarray(pcd_data.normals, dtype=np.float32)
        pcd = BasicPointCloud(points, colors, normals)
        
        if load_vfi:
            return cam_info, pcd, viewpoint_camera, (pcd_vfi, viewpoint_camera_vfi)
        else:
            return cam_info, pcd, viewpoint_camera

    def prepare_data_from_viewpoint(self, idx, down_sample=True,
                                    orthogonal=True, pose=None,
                                    load_depth=True,
                                    load_vfi=False):
        viewpoint_camera = copy(self.data[idx])
        image_name = viewpoint_camera.image_name
        intrinsics = viewpoint_camera.intrinsics
        uid = viewpoint_camera.uid
        if getattr(viewpoint_camera, 'original_image', None) is not None:
            FoVx = viewpoint_camera.FoVx
            FoVy = viewpoint_camera.FoVy
            mask = viewpoint_camera.gt_alpha_mask
            image_torch = viewpoint_camera.original_image
            image_np = image_torch.permute(1, 2, 0).cpu().numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
            w, h = viewpoint_camera.image_width, viewpoint_camera.image_height

        else:
            FoVx = viewpoint_camera.FovX
            FoVy = viewpoint_camera.FovY
            mask = None
            image_pil = Image.open(viewpoint_camera.image_path).convert("RGB")
            image_np = np.asarray(image_pil) / 255.0
            image_torch = torch.from_numpy(
                image_np).permute(2, 0, 1).float()
            w, h = viewpoint_camera.width, viewpoint_camera.height

        cam_info = {}
        

        cam_info["intrinsics"] = intrinsics
        cam_info["FoVx"] = FoVx
        cam_info["FoVy"] = FoVy

        if orthogonal:
            R = np.eye(3)
            t = np.zeros(3)
        elif pose is not None:
            R = pose[:3, :3]
            t = pose[:3, 3]
        else:
            R = np.eye(3)
            t = np.zeros(3)

        if load_depth:
            if idx not in self.mono_depth:
                depth_tensor = self.predict_depth(np.asarray(image_np) * 255)
                depth_tensor[depth_tensor < self.near] = self.near
                self.mono_depth[idx] = depth_tensor.cuda()
            else:
                depth_tensor = self.mono_depth[idx]
        else:
            h, w = image_np.shape[:2]
            depth_tensor = torch.ones((h, w))
            self.mono_depth[idx] = depth_tensor.cuda()

        intr_mat_tensor = torch.from_numpy(
            intrinsics).float().to(depth_tensor.device)
        pts = depth_to_3d(depth_tensor[None, None],
                          intr_mat_tensor[None],
                          normalize_points=False)

        points = pts[0].permute(1, 2, 0).cpu().numpy().reshape(-1, 3)

        viewpoint_camera = Camera(idx, R, t, FoVx, FoVy, image_torch,
                                gt_alpha_mask=mask, image_name=image_name,
                                intrinsics=intrinsics,
                                uid=idx, is_co3d=True,
                                )

        pcd_data = o3d.geometry.PointCloud()
        pcd_data.points = o3d.utility.Vector3dVector(points)
        pcd_data.colors = o3d.utility.Vector3dVector(image_np.reshape(-1, 3))
        pcd_data.estimate_normals()
        if down_sample: # Default: True
            pcd_data = pcd_data.voxel_down_sample(voxel_size=0.01)

        colors = np.asarray(pcd_data.colors, dtype=np.float32)
        points = np.asarray(pcd_data.points, dtype=np.float32)
        normals = np.asarray(pcd_data.normals, dtype=np.float32)

        pcd = BasicPointCloud(points, colors, normals)


        vfi = None
        if load_vfi:
            if idx + 1 < len(self.data):
                if f"{idx}_to_{idx+1}" not in self.vfi:
                    image1 = image_torch.unsqueeze(0).cuda()
                    image2 = copy(self.data[idx+1]).original_image.unsqueeze(0).cuda()

                    vfi = self.predict_vfi(image1, image2)

                    self.vfi[f"{idx}_to_{idx+1}"] = vfi.cuda() # 3, h, w

                else:
                    vfi = self.vfi[f"{idx}_to_{idx+1}"]
            else:
                w, h = image_np.size
                vfi = torch.ones((3, h, w))
                vfi = vfi.cuda()
                self.vfi[idx] = vfi


            depth_tensor_vfi = self.predict_depth(np.asarray(vfi.squeeze().detach().cpu().permute(1, 2, 0).numpy()* 255.0, dtype=np.uint8))

            pts_vfi = depth_to_3d(depth_tensor_vfi[None, None],
                            intr_mat_tensor[None].to(depth_tensor_vfi.device),
                            normalize_points=False)


            points_vfi = pts_vfi[0].permute(1, 2, 0).cpu().reshape(-1, 3)
            if not orthogonal:
                assert False
                pose_inv = torch.inverse(pose)
                points = (pose_inv[:3, :3] @ points.t()).t() + pose_inv[:3, 3]
            points_vfi = points_vfi.numpy()

            colors_vfi = vfi.squeeze().detach().cpu().permute(1, 2, 0).numpy().reshape(-1, 3)
            viewpoint_camera_vfi = Camera(idx + 0.5, R, t, FoVx, FoVy, vfi.squeeze(),
                                    gt_alpha_mask=None, image_name=None,
                                    intrinsics=intrinsics,
                                    uid=idx+0.5, is_co3d=True)

            pcd_data_vfi = o3d.geometry.PointCloud()
            pcd_data_vfi.points = o3d.utility.Vector3dVector(points_vfi)
            pcd_data_vfi.colors = o3d.utility.Vector3dVector(colors_vfi)
            pcd_data_vfi.estimate_normals()
            if down_sample:
                pcd_data_vfi = pcd_data_vfi.voxel_down_sample(voxel_size=0.01)
            colors_vfi = np.asarray(pcd_data_vfi.colors, dtype=np.float32)
            points_vfi = np.asarray(pcd_data_vfi.points, dtype=np.float32)
            normals_vfi = np.asarray(pcd_data_vfi.normals, dtype=np.float32)
            pcd_vfi = BasicPointCloud(points_vfi, colors_vfi, normals_vfi)


        if load_vfi:
            return cam_info, pcd, viewpoint_camera, (pcd_vfi, viewpoint_camera_vfi)
        else:
            return cam_info, pcd, viewpoint_camera
        



    def prepare_data_with_images_only(self, idx, down_sample=True,
                                    orthogonal=True, pose=None,
                                    load_depth=True, load_vfi=False):
        viewpoint_camera = copy(self.data[idx])
        image_name = viewpoint_camera.image_name
        intrinsics = viewpoint_camera.intrinsics
        uid = viewpoint_camera.uid
        if getattr(viewpoint_camera, 'original_image', None) is not None:
            FoVx = viewpoint_camera.FoVx
            FoVy = viewpoint_camera.FoVy
            mask = viewpoint_camera.gt_alpha_mask
            image_torch = viewpoint_camera.original_image
            image_np = image_torch.permute(1, 2, 0).cpu().numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
            w, h = viewpoint_camera.image_width, viewpoint_camera.image_height

        else:
            FoVx = viewpoint_camera.FovX
            FoVy = viewpoint_camera.FovY
            mask = None
            image_pil = Image.open(viewpoint_camera.image_path).convert("RGB")
            image_np = np.asarray(image_pil) / 255.0
            image_torch = torch.from_numpy(
                image_np).permute(2, 0, 1).float()
            w, h = viewpoint_camera.width, viewpoint_camera.height

        cam_info = {}
        cam_info["intrinsics"] = intrinsics
        cam_info["FoVx"] = FoVx
        cam_info["FoVy"] = FoVy

        if orthogonal:
            R = np.eye(3)
            t = np.zeros(3)
        elif pose is not None:
            R = pose[:3, :3]
            t = pose[:3, 3]
        else:
            R = np.eye(3)
            t = np.zeros(3)

        if load_depth:
            if idx not in self.mono_depth:
                depth_tensor = self.predict_depth(np.asarray(image_np) * 255)
                depth_tensor[depth_tensor < self.near] = self.near
                self.mono_depth[idx] = depth_tensor.cuda()
            else:
                depth_tensor = self.mono_depth[idx]
        else:
            h, w = image_np.shape[:2]
            depth_tensor = torch.ones((h, w))
            self.mono_depth[idx] = depth_tensor.cuda()

        intr_mat_tensor = torch.from_numpy(
            intrinsics).float().to(depth_tensor.device)
        pts = depth_to_3d(depth_tensor[None, None],
                          intr_mat_tensor[None],
                          normalize_points=False)

        points = pts[0].permute(1, 2, 0).cpu().numpy().reshape(-1, 3)

        viewpoint_camera = Camera(idx, R, t, FoVx, FoVy, image_torch,
                                  gt_alpha_mask=mask, image_name=image_name,
                                  intrinsics=intrinsics,
                                  uid=idx, is_co3d=True,
                                  )

        pcd_data = o3d.geometry.PointCloud()
        pcd_data.points = o3d.utility.Vector3dVector(points)
        pcd_data.colors = o3d.utility.Vector3dVector(image_np.reshape(-1, 3))
        pcd_data.estimate_normals()
        if down_sample: # Default: True
            pcd_data = pcd_data.voxel_down_sample(voxel_size=0.01)

        colors = np.asarray(pcd_data.colors, dtype=np.float32)
        points = np.asarray(pcd_data.points, dtype=np.float32)
        normals = np.asarray(pcd_data.normals, dtype=np.float32)


        pcd = BasicPointCloud(points, colors, normals)

        vfi = None
        if load_vfi:
            if idx + 1 < len(self.data):
                if f"{idx}_to_{idx+1}" not in self.vfi:
                    image1 = image_torch.unsqueeze(0).cuda()
                    image2 = copy(self.data[idx+1]).original_image.unsqueeze(0).cuda()
                    vfi = self.predict_vfi(image1, image2)
                    self.vfi[f"{idx}_to_{idx+1}"] = vfi.cuda() # 3, h, w
                else:
                    vfi = self.vfi[f"{idx}_to_{idx+1}"]
            else:
                w, h = image_np.size
                vfi = torch.ones((3, h, w))
                vfi = vfi.cuda()
                self.vfi[idx] = vfi

            depth_tensor_vfi = self.predict_depth(np.asarray(vfi.squeeze().detach().cpu().permute(1, 2, 0).numpy()* 255.0, dtype=np.uint8))

            pts_vfi = depth_to_3d(depth_tensor_vfi[None, None],
                            intr_mat_tensor[None].to(depth_tensor_vfi.device),
                            normalize_points=False)


            points_vfi = pts_vfi[0].permute(1, 2, 0).cpu().reshape(-1, 3)
            if not orthogonal:
                assert False
                pose_inv = torch.inverse(pose)
                points = (pose_inv[:3, :3] @ points.t()).t() + pose_inv[:3, 3]
            points_vfi = points_vfi.numpy()

            colors_vfi = vfi.squeeze().detach().cpu().permute(1, 2, 0).numpy().reshape(-1, 3)
            viewpoint_camera_vfi = Camera(idx + 0.5, R, t, FoVx, FoVy, vfi.squeeze(),
                                    gt_alpha_mask=None, image_name=None,
                                    intrinsics=intrinsics,
                                    uid=idx+0.5, is_co3d=True)

            pcd_data_vfi = o3d.geometry.PointCloud()
            pcd_data_vfi.points = o3d.utility.Vector3dVector(points_vfi)
            pcd_data_vfi.colors = o3d.utility.Vector3dVector(colors_vfi)
            pcd_data_vfi.estimate_normals()
            if down_sample:
                pcd_data_vfi = pcd_data_vfi.voxel_down_sample(voxel_size=0.01)
            colors_vfi = np.asarray(pcd_data_vfi.colors, dtype=np.float32)
            points_vfi = np.asarray(pcd_data_vfi.points, dtype=np.float32)
            normals_vfi = np.asarray(pcd_data_vfi.normals, dtype=np.float32)
            pcd_vfi = BasicPointCloud(points_vfi, colors_vfi, normals_vfi)


        if load_vfi:
            return cam_info, pcd, viewpoint_camera, (pcd_vfi, viewpoint_camera_vfi)
        else:
            return cam_info, pcd, viewpoint_camera
        


    def prepare_custom_data(self, idx, down_sample=True,
                            orthogonal=True, pose=None,
                            load_depth=True, **kwargs):
        image_name = self.data[idx]
        intrinsics = self.intrinsic
        uid = idx

        original_image = Image.open(image_name).convert("RGB")
        width, height = original_image.size
        if min(width, height) > 1000:
            original_image = original_image.resize(
                (width // 2, height // 2), Image.LANCZOS)
            width, height = original_image.size
        image_np = np.asarray(original_image) / 255.0
        color_torch = torch.from_numpy(np.asarray(
            original_image) / 255.0).permute(2, 0, 1).float()
        if orthogonal:
            R = np.eye(3)
            t = np.zeros(3)
        elif pose is not None:
            R = pose[:3, :3]
            t = pose[:3, 3]
        else:
            R = np.eye(3)
            t = np.zeros(3)
        focal_length_x = self.intrinsic[0, 0]
        focal_length_y = self.intrinsic[1, 1]
        FoVy = focal2fov(focal_length_y, height)
        FoVx = focal2fov(focal_length_x, width)

        
        cam_info = {}
        pose_src = np.eye(4)
        cam_info["gt_pose"] = copy(pose_src)
        cam_info["intrinsics"] = intrinsics

        cam_info["FoVx"] = FoVx
        cam_info["FoVy"] = FoVy
        cam_info["R"] = R
        cam_info["t"] = t


        if load_depth:
            if idx not in self.mono_depth:
                depth_tensor = self.predict_depth(np.asarray(original_image))
                depth_tensor[depth_tensor < self.near] = self.near
                self.mono_depth[idx] = depth_tensor.cuda()
            else:
                depth_tensor = self.mono_depth[idx]
        else:
            w, h = original_image.size
            depth_tensor = torch.ones((h, w))
            self.mono_depth[idx] = depth_tensor.cuda()

        intr_mat_tensor = torch.from_numpy(
            intrinsics).float().to(depth_tensor.device)
        pts = depth_to_3d(depth_tensor[None, None],
                          intr_mat_tensor[None],
                          normalize_points=False)

        points = pts[0].permute(1, 2, 0).cpu().numpy().reshape(-1, 3)

        viewpoint_camera = Camera(idx, R, t, FoVx, FoVy, color_torch,
                                gt_alpha_mask=None, image_name=image_name,
                                intrinsics=self.intrinsic,
                                uid=idx, is_co3d=True)

        pcd_data = o3d.geometry.PointCloud()
        pcd_data.points = o3d.utility.Vector3dVector(points)
        pcd_data.colors = o3d.utility.Vector3dVector(image_np.reshape(-1, 3))
        pcd_data.estimate_normals()
        if down_sample:
            voxel_size = 0.01
            while len(pcd_data.points)> 1_000_000:
                pcd_data = pcd_data.voxel_down_sample(voxel_size=voxel_size)
                voxel_size *= 5

        colors = np.asarray(pcd_data.colors, dtype=np.float32)
        points = np.asarray(pcd_data.points, dtype=np.float32)
        normals = np.asarray(pcd_data.normals, dtype=np.float32)
        pcd = BasicPointCloud(points, colors, normals)

        return cam_info, pcd, viewpoint_camera


    def prepare_data(self, idx, down_sample=True,
                     orthogonal=True, learn_pose=False,
                     pose=None, load_depth=True,
                     load_gt=False, 
                     load_vfi=False,
                     ):
        if self.data_type == "co3d":
            return self.prepare_data_co3d(idx, down_sample=down_sample,
                                          orthogonal=orthogonal, learn_pose=learn_pose,
                                          pose=pose, load_depth=load_depth,
                                          load_gt=load_gt, load_vfi=load_vfi)
        elif self.data_type == "custom":
            return self.prepare_custom_data(idx, down_sample=down_sample,
                                            orthogonal=orthogonal,
                                            pose=pose,
                                            load_depth=load_depth, load_vfi=load_vfi)
        elif self.data_type == "blender":
            return self.prepare_data_from_viewpoint(idx, down_sample=down_sample,
                                            orthogonal=orthogonal,
                                            pose=pose,
                                            load_depth=load_depth, load_vfi=load_vfi)
        elif self.data_type == "images_only":
            return self.prepare_data_with_images_only(idx, down_sample=down_sample,
                                            orthogonal=orthogonal,
                                            pose=pose,
                                            load_depth=load_depth, load_vfi=load_vfi)
        else:
            return self.prepare_data_from_viewpoint(idx, down_sample=down_sample,
                                                    orthogonal=orthogonal,
                                                    pose=pose,
                                                    load_depth=load_depth,
                                                    load_vfi=load_vfi
                                                    )

    def load_viewpoint_cam(self, idx, pose=None, load_depth=False, pose_to_train=False, load_vfi=False):
        if self.data_type == "co3d":
            image_name = self.data[idx]["image"]["path"]
            image_path = os.path.join(self.data_root, 'co3d', image_name)
            R, t, FoVx, FoVy, intr_mat = self.load_camera(self.data[idx])
            if pose is None:
                R = np.eye(3)
                t = np.zeros(3)
            else:
                R = pose[:3, :3].numpy()
                t = pose[:3, 3].numpy()
            if idx not in self.rgb_images:
                image = Image.open(image_path).convert("RGB")
                self.rgb_images[idx] = image
            else:
                image = self.rgb_images[idx]
            w, h = image.size
            colors_torch = torch.from_numpy(np.asarray(
                image) / 255.0).permute(2, 0, 1).float()


            if load_depth:
                if idx not in self.mono_depth:
                    depth_tensor = self.predict_depth(np.asarray(image))
                    self.mono_depth[idx] = depth_tensor.cuda()

            # Simply copy from 
            vfi = None
            if load_vfi:
                if idx + 1 < len(self.data):
                    if f"{idx}_to_{idx+1}" not in self.vfi:
                        image_path1 = os.path.join(self.data_root, 'co3d', self.data[idx]['image']['path'])
                        image_path2 = os.path.join(self.data_root, 'co3d', self.data[idx + 1]['image']['path'])
                        image1 = imageio.imread(image_path1)
                        image2 = imageio.imread(image_path2)
                        image1 = (torch.tensor(image1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
                        image2 = (torch.tensor(image2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()

                        vfi = self.predict_vfi(image1, image2)

                        self.vfi[f"{idx}_to_{idx+1}"] = vfi.cuda() # 3, h, w

                    else:
                        vfi = self.vfi[f"{idx}_to_{idx+1}"]
                else:
                    w, h = image.size
                    vfi = torch.ones((3, h, w))
                    vfi = vfi.cuda()
                    self.vfi[idx] = vfi
                    
            viewpoint_camera = Camera(idx, R, t, FoVx, FoVy, colors_torch,
                                      gt_alpha_mask=None, image_name=image_name,
                                      intrinsics=intr_mat,
                                      uid=idx, is_co3d=True, vfi=vfi)

        elif self.data_type == "custom":
            image_name = self.data[idx]
            original_image = Image.open(image_name).convert("RGB")
            width, height = original_image.size
            if min(width, height) > 1000:
                original_image = original_image.resize(
                    (width // 2, height // 2), Image.LANCZOS)
                width, height = original_image.size
            color_torch = torch.from_numpy(np.asarray(
                original_image) / 255.0).permute(2, 0, 1).float()
            if pose is None:
                R = np.eye(3)
                t = np.zeros(3)
            else:
                R = pose[:3, :3].numpy()
                t = pose[:3, 3].numpy()
            focal_length_x = self.intrinsic[0, 0]
            focal_length_y = self.intrinsic[1, 1]
            FoVy = focal2fov(focal_length_y, height)
            FoVx = focal2fov(focal_length_x, width)
            viewpoint_camera = Camera(idx, R, t, FoVx, FoVy, color_torch,
                                    gt_alpha_mask=None, image_name=image_name,
                                    intrinsics=self.intrinsic,
                                    uid=idx, is_co3d=True)
            if load_depth:
                if idx not in self.mono_depth:
                    depth_tensor = self.predict_depth(np.asarray(original_image))
                    self.mono_depth[idx] = depth_tensor.cuda()
        else:
            viewpoint_camera = copy(self.data[idx])
            
            if getattr(viewpoint_camera, 'original_image', None) is not None:
                FoVx = viewpoint_camera.FoVx
                FoVy = viewpoint_camera.FoVy
                image_name = viewpoint_camera.image_name
                intrinsics = viewpoint_camera.intrinsics
                uid = viewpoint_camera.uid
                mask = viewpoint_camera.gt_alpha_mask
                original_image = viewpoint_camera.original_image
                image_pil = Image.fromarray((original_image.permute(
                    1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            else:
                FoVx = viewpoint_camera.FovX
                FoVy = viewpoint_camera.FovY
                mask = None
                image_pil = Image.open(
                    viewpoint_camera.image_path).convert("RGB")
                image_np = np.asarray(image_pil) / 255.0
                original_image = torch.from_numpy(
                    image_np).permute(2, 0, 1).float().cuda()
                w, h = viewpoint_camera.width, viewpoint_camera.height
                image_name = viewpoint_camera.image_name
                intrinsics = viewpoint_camera.intrinsics

            if pose is None:
                R = np.eye(3)
                t = np.zeros(3)
            elif pose_to_train is True:
                R = pose[:3, :3]
                t = pose[:3, 3]
            else:
                R = pose[:3, :3].numpy()
                t = pose[:3, 3].numpy()

    

            vfi = None
            if load_vfi:
                if idx + 1 < len(self.data):
                    if f"{idx}_to_{idx+1}" not in self.vfi:
                        image1 = original_image.unsqueeze(0).cuda()
                        image2 = copy(self.data[idx+1]).original_image.unsqueeze(0).cuda()

                        vfi = self.predict_vfi(image1, image2)

                        self.vfi[f"{idx}_to_{idx+1}"] = vfi.cuda() # 3, h, w

                    else:
                        vfi = self.vfi[f"{idx}_to_{idx+1}"]
                else:
                    w, h = image_np.size
                    vfi = torch.ones((3, h, w))
                    vfi = vfi.cuda()
                    self.vfi[idx] = vfi

            viewpoint_camera = Camera(idx, R, t, FoVx, FoVy, original_image,
                                      gt_alpha_mask=None, image_name=image_name,
                                      intrinsics=intrinsics,
                                      uid=idx, is_co3d=True, vfi=vfi)

            if load_depth:
                if idx not in self.mono_depth:
                    depth_tensor = self.predict_depth(np.asarray(image_pil))
                    depth_tensor[depth_tensor < self.near] = self.near
                    self.mono_depth[idx] = depth_tensor.cuda()
            


        return viewpoint_camera

    def train_step(self, viewpoint_cam,
                   iteration, background,
                   pipe, optim_opt, colors_precomp=None):
        # Render
        render_pkg = render(viewpoint_cam, self.model, pipe, background,
                            override_color=colors_precomp)
        image, viewspace_point_tensor, visibility_filter, radii = (render_pkg["render"],
                                                                   render_pkg["viewspace_points"],
                                                                   render_pkg["visibility_filter"],
                                                                   render_pkg["radii"])
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        # loss = (1.0 - optim_opt.lambda_dssim) * Ll1 + optim_opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = Ll1
        loss.backward()

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            psnr_train = psnr(image, gt_image).mean().double()

            if iteration < optim_opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                self.model.max_radii2D[visibility_filter] = torch.max(self.model.max_radii2D[visibility_filter],
                                                                      radii[visibility_filter])
                self.model.add_densification_stats(
                    viewspace_point_tensor, visibility_filter)

                if iteration > optim_opt.densify_from_iter and iteration % optim_opt.densification_interval == 0:
                    size_threshold = 20 if iteration > optim_opt.opacity_reset_interval else None
                    self.model.densify_and_prune(optim_opt.densify_grad_threshold, 0.005,
                                                 self.radius, size_threshold)

                if iteration % optim_opt.opacity_reset_interval == 0:
                    self.model.reset_opacity()
            self.model.optimizer.step()
            self.model.optimizer.zero_grad(set_to_none=True)

        return loss, render_pkg, psnr_train

    def train(self, pipe, optim_opt, viewpoint_cam, colors_precomp=None):
        # fit the following frames
        # _, _, viewpoint_cam = self.prepare_data(idx, orthogonal=True)
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if colors_precomp is not None:
            background = torch.zeros_like(colors_precomp[0])
        progress_bar = tqdm(range(optim_opt.iterations),
                            desc="Training progress")
        self.ema_loss_for_log = 0.0
        # self.model.training_setup(optim_opt)
        for iteration in range(1, optim_opt.iterations):
            # Update learning rate
            self.model.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.model.oneupSHdegree()

            # viewpoint_cam = self.create_viewpoint()
            loss, rend_dict, psnr_train = self.train_step(viewpoint_cam, iteration,
                                                          background, pipe, optim_opt,
                                                          colors_precomp=colors_precomp)

            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}"})
                progress_bar.update(10)
            if iteration == optim_opt.iterations:
                progress_bar.close()

    def obtain_center_feat(self,):
        pass

    def visualize(self, render_pkg, filename):
        if "depth" in render_pkg:
            rend_depth = Image.fromarray(
                colorize(render_pkg["depth"].detach().cpu().numpy(),
                         cmap='magma_r')).convert("RGB")
            rend_depth.save(filename.replace(".png", "_depth.png"))
        if "acc" in render_pkg:
            rend_acc = Image.fromarray(
                colorize(render_pkg["acc"].detach().cpu().numpy(),
                         cmap='magma_r')).convert("RGB")
            rend_acc.save(filename.replace(".png", "_acc.png"))

        rend_img = Image.fromarray(
            np.asarray(render_pkg["render"].detach().cpu().permute(1,
                                                                   2, 0).numpy() * 255.0, dtype=np.uint8)).convert("RGB")
        rend_img.save(filename)
