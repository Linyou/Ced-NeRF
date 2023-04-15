import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from .utils import Rays
from .pose_ulils import correct_poses_bounds

def similarity_from_cameras(c2w, strict_scaling):
    """
    reference: nerf-factory
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))

    return transform, scale

def _load_data_from_json(root_fp, subject_id, factor=1, split='train', read_img=True):

    scene = subject_id

    is_flame_salmon = False
    if 'flame_salmon' in subject_id:
        flame_id = int(subject_id.split('_')[-1])-1
        is_flame_salmon = True
        subject_id = 'flame_salmon_1'

    basedir = os.path.join(root_fp, subject_id)
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    # print("poses_arr: ", poses_arr.shape)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])

    
    json_file = os.path.join(basedir, f'images_x{factor}_list.json')
    with open(json_file) as jf:
        json_data = json.load(jf)
    
    r_w = json_data['videos'][0]['images'][0]['weight']
    r_h = json_data['videos'][0]['images'][0]['height']

    video_list = json_data['videos']
    # scene = json_data['scene']


    poses[:2, 4, :] = np.array([r_h, r_w]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor

    poses = poses.transpose([2,0,1])
    bds = bds.transpose([1,0])

    focal = poses[0, -1, -1]
    HEIGHT = int(poses[0, 0, -1])
    WIDTH = int(poses[0, 1, -1])

    poses, poses_ref, bds = correct_poses_bounds(poses, bds)

    # padding = np.zeros([poses.shape[0], 1, 4])
    # padding[:, 0, 3] = 1
    # poses = np.concatenate([poses[:, :, :4], padding], axis=1)
    # print("poses shape: ", poses.shape)
    # # normalize the scene
    # T, sscale = similarity_from_cameras(
    #     poses, strict_scaling=False
    # )
    # poses = np.einsum("nij, ki -> nkj", poses, T)
    # poses[:, :3, 3] *= sscale

    print("poses shape: ", poses.shape)

    poses[:, :, 1:3] *= -1
    # scale
    pose_radius_scale = 0.4
    poses[:, :, 3] *= pose_radius_scale
    # offset
    poses[:, :, 3] += np.array([[0,0,1.5]])

    if split == 'train':
        load_every = 1
        video_list = video_list[1:]
        poses = poses[1:]
        bds = bds[1:]
    else:
        load_every = 10
        video_list = video_list[:1]
        poses = poses[:1]
        bds = bds[:1]

    images = []
    timestamps = []
    poses_list = []
    bds_list = [] 
    print("loading video:")
    with tqdm(position=0) as progress:
        for i, video in enumerate(video_list):
            v_name = video['video_name']

            pose = poses[i]
            # bd = bds[i]

            vids = video['images']
            if is_flame_salmon:
                vids = vids[flame_id*300:(flame_id+1)*300]
            
            sizeofimage = len(vids)-1 # 0~n-1
            progress.set_description_str(f'{scene}-{v_name}')
            progress.reset(total=len(vids))
            for j, im in enumerate(vids):
                progress.update()
                if j % load_every == 0:
                    idx = im['idx']
                    # images.append(np.array(Image.open(im['path'])).astype(np.uint8)[None, ...])
                    if read_img:
                        images.append(imageio.imread(im['path']).astype(np.uint8))
                    else:
                        images.append(np.array([0,]))
                    timestamps.append(idx/sizeofimage)
                    # timestamps.append(0.)
                    poses_list.append(pose)
                    # bds_list.append(bd)
            progress.refresh()

    images = torch.from_numpy(np.stack(images, axis=0))
    poses_list = np.array(poses_list).astype(np.float32)
    timestamps = np.array(timestamps).astype(np.float32)
    # bds_list = np.array(bds_list).astype(np.float32)
        
    return images, poses_list, timestamps, bds_list, sizeofimage+1, len(video_list), (focal, HEIGHT, WIDTH)


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test"]
    SUBJECT_IDS = [
        "coffee_martini",
        "cook_spinach", 
        "cut_roasted_beef", 
        "flame_salmon_1",
        "flame_salmon_2",
        "flame_salmon_3",
        "flame_salmon_4",
        "flame_steak", 
        "sear_steak"
    ]

    OPENGL_CAMERA = False

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

        (
            self.images,
            self.poses,
            self.timestamps,
            self.bounds,
            self.images_per_video,
            self.num_cameras,
            instrinc
        ) = _load_data_from_json(root_fp, subject_id, factor=factor, split=split, read_img=read_image)

        self.focal, self.HEIGHT, self.WIDTH = instrinc

        # self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.poses[:, :4, :4]).to(torch.float32)
        self.timestamps = torch.from_numpy(self.timestamps).to(torch.float32)[
            :, None
        ]
        

        print("showing a pose: ")
        print(self.poses[0].astype(np.float16))
        # print("showing timestamps: ")
        # print(self.timestamps)
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        if read_image:
            assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)

        self.width, self.height = self.WIDTH, self.HEIGHT

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
            if self.batch_over_images:
                # image_id = torch.randint(
                #     0,
                #     len(self.images),
                #     size=(num_rays,),
                #     device=self.images.device,
                # )
                t_idx = torch.randint(
                    0,
                    self.images_per_video,
                    size=(num_rays,),
                    device=self.images.device,
                )
                cam_id = torch.randint(
                    0,
                    self.num_cameras,
                    size=(num_rays,),
                    device=self.images.device,
                )
                image_id = cam_id * self.images_per_video + t_idx
            else:
                image_id = [index]
            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.images.device),
                torch.arange(self.HEIGHT, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgb = self.images[image_id, y, x] / 255.0  # (num_rays, 3)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5) / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
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

        rays = Rays(
            origins=origins, 
            viewdirs=viewdirs
        )
        timestamps = self.timestamps[image_id]

        return {
            "rgb": rgb,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "timestamps": timestamps,  # [num_rays, 1]
            "idx": image_id,
        }


    def get_ltrb(self, index):
        """Get the left, top, right, bottom of the image."""
        x, y = torch.tensor([0, self.width, self.width, 0]), torch.tensor([0, 0, self.height, self.height])
        image_id = [index]
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape).numpy()
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        ).numpy()
        empty_dir = np.array([[0, 0, 0]]).astype(np.float32)
        pad_origins = np.concatenate([origins[:1], origins], axis=0)
        pad_viewdirs = np.concatenate([empty_dir, viewdirs], axis=0)

        timestamps = self.timestamps[image_id]

        return pad_origins[:, [0, 1, 2]], pad_viewdirs[:, [0, 1, 2]], timestamps.numpy()

    def get_rays(self, index, device, ext_func):
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=device,
                )
            else:
                image_id = [index]
            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device=device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=device),
                torch.arange(self.HEIGHT, device=device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5) / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        c2w = ext_func(c2w)

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
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

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return rays