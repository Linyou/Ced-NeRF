import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm, trange
from .utils import Rays, generate_spiral_path
from .pose_ulils import correct_poses_bounds, create_spiral_poses

@torch.no_grad()
def dynerf_isg_weight(imgs, median_imgs, gamma):
    # imgs is [num_cameras * num_frames, h, w, 3]
    # median_imgs is [num_cameras, h, w, 3]
    assert imgs.dtype == torch.uint8
    assert median_imgs.dtype == torch.uint8
    num_cameras, h, w, c = median_imgs.shape
    squarediff = (
        imgs.view(num_cameras, -1, h, w, c)
            .float()  # creates new tensor, so later operations can be in-place
            .div_(255.0)
            .sub_(
                median_imgs[:, None, ...].float().div_(255.0)
            )
            .square_()  # noqa
    )  # [num_cameras, num_frames, h, w, 3]
    # differences = median_imgs[:, None, ...] - imgs.view(num_cameras, -1, h, w, c)  # [num_cameras, num_frames, h, w, 3]
    # squarediff = torch.square_(differences)
    psidiff = squarediff.div_(squarediff + gamma**2)
    psidiff = (1./3) * torch.sum(psidiff, dim=-1)  # [num_cameras, num_frames, h, w]
    return psidiff  # valid probabilities, each in [0, 1]


@torch.no_grad()
def dynerf_ist_weight(imgs, num_cameras, alpha=0.1, frame_shift=25):  # DyNerf uses alpha=0.1
    assert imgs.dtype == torch.uint8
    N, h, w, c = imgs.shape
    frames = imgs.view(num_cameras, -1, h, w, c).float()  # [num_cameras, num_timesteps, h, w, 3]
    max_diff = None
    shifts = list(range(frame_shift + 1))[1:]
    for shift in shifts:
        shift_left = torch.cat([frames[:, shift:, ...], torch.zeros(num_cameras, shift, h, w, c)], dim=1)
        shift_right = torch.cat([torch.zeros(num_cameras, shift, h, w, c), frames[:, :-shift, ...]], dim=1)
        mymax = torch.maximum(torch.abs_(shift_left - frames), torch.abs_(shift_right - frames))
        if max_diff is None:
            max_diff = mymax
        else:
            max_diff = torch.maximum(max_diff, mymax)  # [num_timesteps, h, w, 3]
        
    max_diff = torch.mean(max_diff, dim=-1)  # [num_timesteps, h, w]
    max_diff = max_diff.clamp_(min=alpha)
    return max_diff

@torch.no_grad()
def dynerf_ist_weight_nice(imgs, num_cameras, alpha=0.1, frame_shift=25):
    assert imgs.dtype == torch.uint8
    N, h, w, c = imgs.shape
    frames = imgs.view(num_cameras, -1, h, w, c).float()
    max_diff_list = []

    shifts = list(range(frame_shift + 1))[1:]
    print("loop over cameras")
    for cam_id in trange(num_cameras):
        max_diff_cam = torch.zeros(N//num_cameras, h, w, c)
        for shift in shifts:
            shift_left = torch.cat([frames[cam_id, shift:, ...], torch.zeros(shift, h, w, c)], dim=0)
            shift_right = torch.cat([torch.zeros(shift, h, w, c), frames[cam_id, :-shift, ...]], dim=0)
            mymax = torch.maximum(torch.abs_(shift_left - frames[cam_id]), torch.abs_(shift_right - frames[cam_id]))
            max_diff_cam = torch.maximum(max_diff_cam, mymax)
            del mymax, shift_left, shift_right
            torch.cuda.empty_cache()
        max_diff_list.append(torch.mean(max_diff_cam, dim=-1).clamp_(min=alpha))
    max_diff = torch.stack(max_diff_list)
    return max_diff

def _load_data_from_json(root_fp, subject_id, factor=1, split='train', read_img=True, load_every=1):

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
    # print("bds: ", bds)
    render_poses = generate_spiral_path(
        poses[:, :3, :4], 
        bds, 
        n_frames=300,
        n_rots=2, 
        zrate=0.1, 
        dt=0.7, 
        percentile=50,
    )


    print("poses shape: ", poses.shape)

    poses[:, :, 1:3] *= -1
    render_poses[:, :, 1:3] *= -1
    # scale
    pose_radius_scale = 0.4
    poses[:, :, 3] *= pose_radius_scale
    render_poses[:, :, 3] *= pose_radius_scale
    # offset
    poses[:, :, 3] += np.array([[0,0,1.5]])
    render_poses[:, :, 3] += np.array([[0,0,1.5]])

    if split == 'train':
        video_list = video_list[1:]
        poses = poses[1:]
        bds = bds[1:]
    else:
        video_list = video_list[:1]
        poses = poses[:1]
        bds = bds[:1]

    images = []
    med_imgs = []
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
            images_per_cam = []
            for j, im in enumerate(vids):
                progress.update()
                if j % load_every == 0:
                    idx = im['idx']
                    # images.append(np.array(Image.open(im['path'])).astype(np.uint8)[None, ...])
                    if read_img:
                        img_path = os.path.join(basedir, im['path'])
                        images_per_cam.append(imageio.imread(img_path).astype(np.uint8))
                    else:
                        images_per_cam.append(np.array([0,]))
                    timestamps.append(idx/sizeofimage)
                    # timestamps.append(0.)
                    poses_list.append(pose)
                    # bds_list.append(bd)

            med_img, _ = torch.median(torch.from_numpy(np.stack(images_per_cam, 0)), dim=0)  # [h, w, 3]
            med_imgs.append(med_img)
            images += images_per_cam
            progress.refresh()

    images = torch.from_numpy(np.stack(images, axis=0))
    median_imgs = torch.stack(med_imgs, 0)
    poses_list = np.array(poses_list).astype(np.float32)
    timestamps = np.array(timestamps).astype(np.float32)
    # bds_list = np.array(bds_list).astype(np.float32)
        
    return images, poses_list, timestamps, bds_list, sizeofimage+1, len(video_list), (focal, HEIGHT, WIDTH), render_poses, median_imgs


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
            instrinc,
            render_poses,
            self.median_imgs,
        ) = _load_data_from_json(
            root_fp, 
            subject_id, 
            factor=factor, 
            split=split, 
            read_img=read_image,
            load_every=1 if split == "train" else 10,
        )


        basedir = os.path.join(root_fp, subject_id)
        if os.path.exists(os.path.join(basedir, f"isg_weights.pt")):
            self.isg_weights = torch.load(os.path.join(basedir, f"isg_weights.pt"))
            print(f"Reloaded {self.isg_weights.shape[0]} ISG weights from file.")

        if os.path.exists(os.path.join(basedir, f"ist_weights.pt")):
            self.ist_weights = torch.load(os.path.join(basedir, f"isg_weights.pt"))
            print(f"Reloaded {self.ist_weights.shape[0]} IST weights from file.")

        self.focal, self.HEIGHT, self.WIDTH = instrinc

        # self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.poses[:, :4, :4]).to(torch.float32)
        self.timestamps = torch.from_numpy(self.timestamps).to(torch.float32)[
            :, None
        ]
        self.render_poses = torch.from_numpy(render_poses).to(torch.float32)

        print("render_poses: ", self.render_poses.shape)
        print(self.render_poses[0])
        

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

        self.weights_subsampled = int(4 / factor)
        self.sampling_weights = self.isg_weights
        self.sampling_batch_size = 2_000_000

    def switch_to_ist(self):
        self.sampling_weights = self.ist_weights

    def __len__(self):
        return len(self.camtoworlds)

    def to(self, device):
        self.K = self.K.to(device)
        self.images = self.images.to(device)
        self.camtoworlds = self.camtoworlds.to(device)
        self.timestamps = self.timestamps.to(device)
        self.render_poses = self.render_poses.to(device)
        return self

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data
    
    def get_render_poses(self, index):
        image_id = [index]
        x, y = torch.meshgrid(
            torch.arange(self.WIDTH, device=self.images.device),
            torch.arange(self.HEIGHT, device=self.images.device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()

        # generate rays
        c2w = self.render_poses[image_id]  # (num_rays, 3, 4)
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
        origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
        viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
        directions = torch.reshape(directions, (self.HEIGHT, self.WIDTH, 3))
        rays = Rays(
            origins=origins, 
            viewdirs=viewdirs
        )
        timestamps = torch.tensor([[index/self.render_poses.shape[0]]])

        return {
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "timestamps": timestamps,  # [num_rays, 1]
        }

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
            batch_size = self.num_rays // (self.weights_subsampled ** 2)
            num_weights = len(self.sampling_weights)
            if num_weights > self.sampling_batch_size:
                # Take a uniform random sample first, then according to the weights
                subset = torch.randint(
                    0, num_weights, size=(self.sampling_batch_size,),
                    dtype=torch.int64, device=self.sampling_weights.device)
                samples = torch.multinomial(
                    input=self.sampling_weights[subset], num_samples=batch_size)
                index = subset[samples]
            else:
                index = torch.multinomial(
                    input=self.sampling_weights, num_samples=batch_size)
                

            # We must deal with the fact that ISG/IST weights are computed on a dataset with
            # different 'downsampling' factor. E.g. if the weights were computed on 4x
            # downsampled data and the current dataset is 2x downsampled, `weights_subsampled`
            # will be 4 / 2 = 2.
            # Split each subsampled index into its 16 components in 2D.
            hsub, wsub = self.height // self.weights_subsampled, self.width // self.weights_subsampled
            image_id = torch.div(index, hsub * wsub, rounding_mode='floor')
            ysub = torch.remainder(index, hsub * wsub).div(wsub, rounding_mode='floor')
            xsub = torch.remainder(index, hsub * wsub).remainder(wsub)
            # xsub, ysub is the first point in the 4x4 square of finely sampled points
            x, y = [], []
            for ah in range(self.weights_subsampled):
                for aw in range(self.weights_subsampled):
                    x.append(xsub * self.weights_subsampled + aw)
                    y.append(ysub * self.weights_subsampled + ah)
            x = torch.cat(x)
            y = torch.cat(y)
            image_id = image_id.repeat(self.weights_subsampled ** 2)
            # Inverse of the process to get x, y from index. image_id stays the same.
            index = x + y * self.width + image_id * self.height * self.width
            
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

