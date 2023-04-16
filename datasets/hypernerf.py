import warnings

warnings.filterwarnings("ignore")

import json
import os
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from .utils import Rays
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm
import pdb


def _compute_residual_and_jacobian(
    x: torch.Tensor,
    y: torch.Tensor,
    xd: torch.Tensor,
    yd: torch.Tensor,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
  """Auxiliary function of radial_and_tangential_undistort()."""

  r = x * x + y * y
  d = 1.0 + r * (k1 + r * (k2 + k3 * r))

  fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
  fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

  # Compute derivative of d over [x, y]
  d_r = (k1 + r * (2.0 * k2 + 3.0 * k3 * r))
  d_x = 2.0 * x * d_r
  d_y = 2.0 * y * d_r

  # Compute derivative of fx over x and y.
  fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
  fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

  # Compute derivative of fy over x and y.
  fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
  fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

  return fx, fy, fx_x, fx_y, fy_x, fy_y

def _radial_and_tangential_undistort(
    xd: torch.Tensor,
    yd: torch.Tensor,
    distortion: torch.Tensor,
    eps: float = 1e-9,
    max_iterations=10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes undistorted (x, y) from (xd, yd)."""
    # Initialize from the distorted point.
    x = xd.clone()
    y = yd.clone()

    k1, k2, k3, p1, p2 = distortion

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(
            torch.abs(denominator) > eps, x_numerator / denominator,
            torch.zeros_like(denominator))
        step_y = torch.where(
            torch.abs(denominator) > eps, y_numerator / denominator,
            torch.zeros_like(denominator))

    x = x + step_x
    y = y + step_y

    return x, y

class Load_hyper_data():
    def __init__(self, 
                 datadir, 
                 ratio=0.5,
                 use_bg_points=False,
                 add_cam=False):
        from .hyper_cam import Camera
        datadir = os.path.expanduser(datadir)
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)

        self.near = scene_json['near']
        self.far = scene_json['far']
        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']

        self.all_img = dataset_json['ids']
        self.val_id = dataset_json['val_ids']

        self.add_cam = False
        if len(self.val_id) == 0:
            self.i_train = np.array([i for i in np.arange(len(self.all_img)) if
                            (i%4 == 0)])
            self.i_test = self.i_train+2
            self.i_test = self.i_test[:-1,]
        else:
            self.add_cam = True
            self.train_id = dataset_json['train_ids']
            self.i_test = []
            self.i_train = []
            for i in range(len(self.all_img)):
                id = self.all_img[i]
                if id in self.val_id:
                    self.i_test.append(i)
                if id in self.train_id:
                    self.i_train.append(i)
        assert self.add_cam == add_cam
        
        # print('self.i_train',self.i_train)
        # print('self.i_test',self.i_test)
        self.all_cam = [meta_json[i]['camera_id'] for i in self.all_img]
        self.all_time = [meta_json[i]['time_id'] for i in self.all_img]
        max_time = max(self.all_time)
        self.all_time = [meta_json[i]['time_id']/max_time for i in self.all_img]
        self.selected_time = set(self.all_time)
        self.ratio = ratio


        # all poses
        self.all_cam_params = []
        for im in self.all_img:
            camera = Camera.from_json(f'{datadir}/camera/{im}.json')
            camera = camera.scale(ratio)
            camera.position = camera.position - self.scene_center
            camera.position = camera.position * self.coord_scale
            self.all_cam_params.append(camera)

        self.all_img = [f'{datadir}/rgb/{int(1/ratio)}x/{i}.png' for i in self.all_img]
        self.h, self.w = self.all_cam_params[0].image_shape

        self.use_bg_points = use_bg_points
        if use_bg_points:
            with open(f'{datadir}/points.npy', 'rb') as f:
                points = np.load(f)
            self.bg_points = (points - self.scene_center) * self.coord_scale
            self.bg_points = torch.tensor(self.bg_points).float()
        print(f'total {len(self.all_img)} images ',
                'use cam =',self.add_cam, 
                'use bg_point=',self.use_bg_points)

    def load_idx(self, idx,not_dic=False):

        all_data = self.load_raw(idx)
        if not_dic == True:
            rays_o = all_data['rays_ori']
            rays_d = all_data['rays_dir']
            viewdirs = all_data['viewdirs']
            rays_color = all_data['rays_color']
            return rays_o, rays_d, viewdirs,rays_color
        return all_data

    def load_raw(self, idx):
        image = Image.open(self.all_img[idx])
        camera = self.all_cam_params[idx]
        pixels = camera.get_pixel_centers()
        rays_dir = torch.tensor(camera.pixels_to_rays(pixels)).float().view([-1,3])
        rays_ori = torch.tensor(camera.position[None, :]).float().expand_as(rays_dir)
        rays_color = torch.tensor(np.array(image)).view([-1,3])/255.
        return {'rays_ori': rays_ori, 
                'rays_dir': rays_dir, 
                'viewdirs':rays_dir / rays_dir.norm(dim=-1, keepdim=True),
                'rays_color': rays_color, 
                'near': torch.tensor(self.near).float().view([-1]), 
                'far': torch.tensor(self.far).float().view([-1]),}
    

def _load_data_from_json(
        datadir, 
        ratio=0.5, 
        use_bg_points=False, 
        add_cam=False,
        split='train',
        read_image=True
    ):
    hyper_data = Load_hyper_data(
        datadir=datadir,
        ratio=ratio,
        use_bg_points=use_bg_points,
        add_cam=add_cam
    )

    # fx = hyper_data.all_cam_params[0].scale_factor_x.item()
    # fy = hyper_data.all_cam_params[0].scale_factor_y.item()
    h, w = hyper_data.all_cam_params[0].image_shape

    img_hw = (h.item(), w.item())
    # k1, k2, k3 = hyper_data.all_cam_params[0].radial_distortion
    # p1, p2 = hyper_data.all_cam_params[0].tangential_distortion
    # extra_info = (k1, k2, k3, p1, p2)

    images = []
    poses = []
    timestamps = []
    focals = []
    distortions = []

    cam_data = []

    if split == 'train':
        iter_idx = hyper_data.i_train
    elif split == 'test':
        iter_idx = hyper_data.i_test

    for i in iter_idx:
        # load image
        if read_image:
            images.append(imageio.imread(hyper_data.all_img[i]).astype(np.uint8))
        else:
            images.append(np.array([0,]))

        # load pose
        R = hyper_data.all_cam_params[i].orientation.T
        T = hyper_data.all_cam_params[i].position[:, None]
        c2w = np.concatenate([R, T], axis=-1)
        poses.append(c2w)

        # extra info
        fx = hyper_data.all_cam_params[i].scale_factor_x.item()
        fy = hyper_data.all_cam_params[i].scale_factor_y.item()
        focals.append(np.array([fx, fy]))

        distortions.append(
            np.concatenate([
                hyper_data.all_cam_params[i].radial_distortion,
                hyper_data.all_cam_params[i].tangential_distortion
            ])
        )

        # load timestamp
        timestamps.append(hyper_data.all_time[i])
        cam_data.append(hyper_data.all_cam_params[i])

    images = torch.from_numpy(np.stack(images, axis=0))
    poses = np.array(poses).astype(np.float32)
    # poses[:, :, 1:3] *= -1
    # poses[:, :, 0] *= -1
    # poses[:, :, 1] *= -1
    # poses[:, :, 2] *= -1
    # poses[:, 0, 3] *= -1
    # poses[:, 2, 3] *= -1
    # scale
    pose_radius_scale = 1.
    poses[:, :, 3] *= pose_radius_scale
    # # offset
    poses[:, :, 3] += np.array([[0, 0 ,0.0]])
    timestamps = np.array(timestamps).astype(np.float32)

    focals = np.array(focals).astype(np.float32)
    distortions = np.array(distortions).astype(np.float32)

    return images, poses, timestamps, focals, distortions, img_hw, cam_data
            


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test"]
    SUBJECT_IDS = [
        'interp_aleks-teapot',
        'interp_chickchicken',
        'interp_cut-lemon',
        'interp_hand',
        'interp_slice-banana',
        'interp_torchocolate',
        'misc_americano',
        'misc_cross-hands',
        'misc_espresso',
        'misc_keyboard',
        'misc_oven-mitts',
        'misc_split-cookie',
        'misc_tamping',
        'vrig_3dprinter',
        'vrig_broom',
        'vrig_chicken',
        'vrig_peel-banana',
    ]
    SUB_SPLIT = [
        'interp_',
        'misc_',
        'vrig_',
    ]

    OPENGL_CAMERA = True

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        factor: int = 1,
        read_image=True,
        add_cam=False,
        device: str = "cpu",
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images

        # get sub split
        for sub_split in self.SUB_SPLIT:
            if subject_id.startswith(sub_split):
                self.sub_split = sub_split
        (
            self.images,
            self.poses,
            self.timestamps,
            self.focals,
            self.distortions,
            img_hw,
            self.hyper_data
        ) = _load_data_from_json(
            os.path.join(
                root_fp, 
                subject_id, 
                subject_id.split(self.sub_split)[-1]
            ), 
            ratio=1/factor, 
            split=split, 
            read_image=read_image,
            add_cam=add_cam
        )
        self.HEIGHT, self.WIDTH = img_hw

        # self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.poses[:, :4, :4]).to(torch.float32)
        self.timestamps = torch.from_numpy(self.timestamps).to(torch.float32)[
            :, None
        ]
        # self.focals = torch.from_numpy(self.focals).to(torch.float32)
        # self.distortions = torch.from_numpy(self.distortions).to(torch.float32)
        
        self.principal_point_x = self.WIDTH / 2.
        self.principal_point_y = self.HEIGHT / 2.

        print("showing a pose: ")
        print(self.poses[0].astype(np.float16))
        # print("showing timestamps: ")
        # print(self.timestamps)
        # print("focal_x: ", self.focal_x)
        # print("focal_y: ", self.focal_y)
        # print("height: ", self.HEIGHT)
        # print("width: ", self.WIDTH)
        Ks = []
        for f in tqdm(self.focals):
            temp_K = torch.eye(3)
            temp_K[0, 0] = f[0].item()
            temp_K[1, 1] = f[1].item()
            temp_K[0, 2] = self.principal_point_x
            temp_K[1, 2] = self.principal_point_y
            Ks.append(temp_K)

        self.K = torch.stack(Ks)
        # self.K = torch.tensor(
        #     [
        #         [self.focal_x, 0, self.principal_point_x],
        #         [0, self.focal_y, self.principal_point_y],
        #         [0, 0, 1],
        #     ],
        #     dtype=torch.float32,
        # )  # (N, 3, 3)
        if read_image:
            assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)

        self.width, self.height = self.WIDTH, self.HEIGHT
        print(f"image width: {self.width}, height: {self.height}")

    def __len__(self):
        return len(self.images)

    def to(self, device):
        self.K = self.K.to(device)
        self.images = self.images.to(device)
        self.camtoworlds = self.camtoworlds.to(device)
        self.timestamps = self.timestamps.to(device)
        return self

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        pixels, rays = data["rgb"], data["rays"]

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            if self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgb", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            # if self.batch_over_images:
            #     image_id = torch.randint(
            #         0,
            #         len(self.images),
            #         size=(num_rays,),
            #         device=self.images.device,
            #     )
            # else:
            image_id = torch.randint(
                0,
                len(self.images),
                size=(1,),
                device=self.images.device,
            ).expand(num_rays).clone()
            # image_id = torch.tensor(
            #     [index], device=self.images.device
            # ).expand(num_rays).clone()
                # image_id = [index]
            # x = torch.randint(
            #     0, self.WIDTH, size=(num_rays,), device=self.images.device
            # )
            # y = torch.randint(
            #     0, self.HEIGHT, size=(num_rays,), device=self.images.device
            # )
            x = torch.randint(
                0, self.WIDTH, size=(num_rays,)
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,)
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH),
                torch.arange(self.HEIGHT),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()


        # generate rays
        rgb = self.images[image_id, y, x] / 255.0  # (num_rays, 3)
        # c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        # x = (x - self.principal_point_x + 0.5) / self.focals[image_id[0], 0] 
        # y = (y - self.principal_point_y + 0.5) / self.focals[image_id[0], 1] 
        # x, y = _radial_and_tangential_undistort(
        #     x, y,
        #     self.distortions[image_id[0]],
        # )
        # # print('distortions: ', self.distortions[image_id[0]])
        # camera_dirs = F.pad(
        #     torch.stack(
        #         [
        #             x,
        #             y * (-1.0 if self.OPENGL_CAMERA else 1.0),
        #         ],
        #         dim=-1,
        #     ),
        #     (0, 1),
        #     value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        # )  # [num_rays, 3]

        # # [n_cams, height, width, 3]
        # directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        # origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        pix = np.stack([x, y], axis=-1) + 0.5
        camera = self.hyper_data[image_id[0]]
        directions = torch.tensor(camera.pixels_to_rays(pix.astype(np.float32))).float().view([-1,3])
        origins = torch.tensor(camera.position[None, :]).float().expand_as(directions)

        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            directions = torch.reshape(directions, (num_rays, 3))
            rgb = torch.reshape(rgb, (num_rays, 3))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            directions = torch.reshape(directions, (self.HEIGHT, self.WIDTH, 3))
            rgb = torch.reshape(rgb, (self.HEIGHT, self.WIDTH, 3))

        rays = Rays(origins=origins, viewdirs=directions)
        timestamps = self.timestamps[image_id]

        return {
            "rgb": rgb,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "timestamps": timestamps,  # [num_rays, 1]
            "idx": image_id,
        }
    