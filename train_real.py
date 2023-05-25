"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import pathlib
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from gui import GUI
from cednerf.model import DNGPradianceField
from cednerf.losses import distortion
from pytorch_msssim import ms_ssim


from cednerf.utils import (
    render_image,
    render_image_test,
    set_random_seed,
)

from nerfacc.estimators.occ_grid import OccGridEstimator
import itertools
from datasets import (
    DNERF_SYNTHETIC_SCENES,
    DYNERF_SCENES,
    HYPERNERF_SCENES
)
from datasets.utils import namedtuple_map

from opt import get_model_args
import cv2
def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).cpu().numpy().astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    # default=str(pathlib.Path.cwd() / "data/360_v2"),
    default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)

parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    choices=DNERF_SYNTHETIC_SCENES + DYNERF_SCENES + HYPERNERF_SCENES,
    help="which scene to use",
)

parser.add_argument(
    "--gui",
    action="store_true",
    help="whether to use GUI for visualization",
)

parser = get_model_args(parser)

args = parser.parse_args()

print(args.data_root)

device = "cuda:0"
set_random_seed(42)

if args.scene in DNERF_SYNTHETIC_SCENES:
    from datasets.dnerf_synthetic import SubjectLoader
    # training parameters
    max_steps = 20000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = 0.0
    # weight_decay = (
    #     1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    # )
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])
    near_plane = 0.0
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    # model parameters
    moving_step = 0.0001
    hash_dst_resolution = 1024
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    alpha_thre = 0.0
    cone_angle = 0.0
    milestones=[
        max_steps // 2,
        max_steps * 3 // 4,
        # max_steps * 5 // 6,
        max_steps * 9 // 10,
    ]

elif args.scene in HYPERNERF_SCENES:
    from datasets.hypernerf import SubjectLoader

    # training parameters
    max_steps = 20000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = 0.0
    # scene parameters
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    near_plane = 0.2
    far_plane = 1.0e10
    # dataset parameters
    add_cam = True if 'vrig' in args.scene else False
    train_dataset_kwargs = {"color_bkgd_aug": "black", "factor": 2, "add_cam": add_cam}
    test_dataset_kwargs = {"color_bkgd_aug": "black", "factor": 2, "add_cam": add_cam}
    # model parameters
    moving_step = 1/4096
    hash_dst_resolution = 4096
    grid_resolution = 128
    grid_nlvl = 2
    # render parameters
    render_step_size = 1e-3
    alpha_thre = 1e-2
    cone_angle = 0.004
    milestones=[
        max_steps // 2,
        max_steps * 3 // 4,
        # max_steps * 5 // 6,
        max_steps * 9 // 10,
    ]

else:
    from datasets.dnerf_3d_video import SubjectLoader

    # training parameters
    max_steps = 20000
    init_batch_size = 8192
    target_sample_batch_size = 1 << 18
    weight_decay = 0.0
    # scene parameters
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    near_plane = 0.2
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
    test_dataset_kwargs = {"color_bkgd_aug": "black", "factor": 4}
    # model parameters
    moving_step = 1/4096
    hash_dst_resolution = 4096
    grid_resolution = 128
    grid_nlvl = 5
    # render parameters
    render_step_size = 1e-3
    alpha_thre = 1e-2
    cone_angle = 0.004
    milestones=[
        max_steps // 2,
        max_steps * 3 // 4,
        # max_steps * 5 // 6,
        max_steps * 9 // 10,
    ]


estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl,
)

if args.load_model:
    max_steps = -1
else:
    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split=args.train_split,
        num_rays=init_batch_size,
        **train_dataset_kwargs,
    )
    if args.scene in DNERF_SYNTHETIC_SCENES:
        train_dataset = train_dataset.to(device)

    if args.scene in DYNERF_SCENES and args.gui:
        estimator = estimator.to(device)
        mark_invisible_K = train_dataset.K
        estimator.mark_invisible_cells(
            mark_invisible_K.unsqueeze(0).clone().to(device), 
            train_dataset.camtoworlds.clone().to(device), 
            train_dataset.width, 
            train_dataset.height,
            near_plane,
        )
    else:
        estimator = estimator.to(device)
        mark_invisible_K = train_dataset.K

    # data_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     num_workers=16,
    #     batch_size=None,
    # )
    # data_loader = itertools.cycle(data_loader)


estimator = estimator.to(device)
test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    **test_dataset_kwargs,
)
if args.scene in DNERF_SYNTHETIC_SCENES:
    test_dataset = test_dataset.to(device)

# if args.scene in HYPERNERF_SCENES and GUI:
#     idx = torch.randint(0, len(train_dataset.K), (1,)).item()
#     mark_invisible_K = train_dataset.K[idx]
#     estimator.mark_invisible_cells(
#         train_dataset.K[idx][None], 
#         train_dataset.camtoworlds, 
#         train_dataset.width, 
#         train_dataset.height,
#         near_plane,
#     )
#     # call mark_invisible_cells before sending the estimator to device.
#     estimator = estimator.to(device)
# el



# setup the radiance field we want to train.
grad_scaler = torch.cuda.amp.GradScaler()
radiance_field = DNGPradianceField(
    aabb=estimator.aabbs[-1],
    moving_step=moving_step,
    dst_resolution=hash_dst_resolution,
    use_div_offsets=args.use_div_offsets,
    use_time_embedding=args.use_time_embedding,
    use_time_attenuation=args.use_time_attenuation,
    use_feat_predict=args.use_feat_predict,
    use_weight_predict=args.use_weight_predict,
    # hash4motion=True,
    # time_inject_before_sigma=False,
).to(device)

try:
    import apex
    optimizer = apex.optimizers.FusedAdam(radiance_field.parameters(), lr=1e-2, eps=1e-15)
except ImportError:
    print("Failed to import apex FusedAdam, use torch Adam instead.")
    optimizer = torch.optim.Adam(
        radiance_field.parameters(), lr=1e-2, eps=1e-15, weight_decay=weight_decay
    )

scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.33,
        ),
    ]
)


# training
tic = time.time()
# pre-set the len of train_dataloader to max_steps,
# so that we can use just iterate over it.
half_steps = max_steps // 2
for step in range(max_steps + 1):
    radiance_field.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]
    # data = next(data_loader)
    
    if args.scene in DNERF_SYNTHETIC_SCENES:
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]
        timestamps = data["timestamps"]
    else:
        render_bkgd = data["color_bkgd"].to(device)
        rays = namedtuple_map(lambda r: r.to(device), data["rays"])
        pixels = data["pixels"].to(device)
        timestamps = data["timestamps"].to(device)

    def occ_eval_fn(x):
        t_idxs = torch.randint(0, len(timestamps), (x.shape[0],), device=x.device)
        t = timestamps[t_idxs]
        density = radiance_field.query_density(x, t)['density']
        return density * render_step_size
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        # update occupancy grid
        estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )

        # render
        rgb, acc, depth, n_rendering_samples, extra = render_image(
            radiance_field,
            estimator,
            rays,
            # rendering options
            near_plane=near_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
            timestamps=timestamps,
        )
        if n_rendering_samples == 0:
            continue

        if target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays * (target_sample_batch_size / float(n_rendering_samples))
            )
            train_dataset.update_num_rays(num_rays)

        # if args.scene in DNERF_SYNTHETIC_SCENES:
        # alive_ray_mask = acc.squeeze(-1) > 0
        # rgb = rgb[alive_ray_mask]
        # pixels = pixels[alive_ray_mask]
        # acc = acc[alive_ray_mask]

        # compute loss
        loss = F.mse_loss(rgb, pixels)

        if args.use_opacity_loss:
            loss += (-acc*torch.log(acc)).mean()*1e-3

        for interal_data in extra:
        
            if args.distortion_loss:
                loss += distortion(
                    interal_data['ray_indices'], 
                    interal_data['weights'], 
                    interal_data['t_starts'], 
                    interal_data['t_ends']
                ) * 1e-3

            if args.acc_entorpy_loss:
                T_last = 1 - acc
                T_last = T_last.clamp(1e-6, 1-1e-6)
                entropy_loss = -(T_last*torch.log(T_last) + (1-T_last)*torch.log(1-T_last)).mean()
                loss += entropy_loss*1e-3

            if args.weight_rgbper:
                rgbper = (interal_data['rgbs'] - pixels[interal_data['ray_indices']]).pow(2).sum(dim=-1)
                loss += (rgbper * interal_data['weights'].detach()).sum() / pixels.shape[0] * 1e-3

            if args.use_feat_predict:
                # if args.scene in DNERF_SYNTHETIC_SCENES:
                    # loss += interal_data['latent_losses'][alive_ray_mask].mean()
                # else:
                    loss += interal_data['latent_losses'].mean()

            if args.use_weight_predict:
                loss += interal_data['weight_losses'].mean()

    
    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()
    # optimizer.step()
    # scheduler.step()
    grad_scaler.step(optimizer)
    scale = grad_scaler.get_scale()
    grad_scaler.update()
    scheduler.step()

    if step % 10000 == 0:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )

    if step > 0 and step % max_steps == 0:

        torch.save(
            {
                "radiance_field": radiance_field.state_dict(),
                "occupancy_grid": estimator.state_dict(),
            },
            "model.pth",
        )

        # evaluation
        radiance_field.eval()
        estimator.eval()

        psnrs = []
        lpips = []
        ssims = []
        with torch.no_grad():
            progress_bar = tqdm.tqdm(total=len(test_dataset), desc=f'evaluating: ')

            for test_step in range(len(test_dataset)):
                progress_bar.update()

                data = test_dataset[test_step]
                if args.scene in DNERF_SYNTHETIC_SCENES:
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    pixels = data["pixels"]
                    timestamps = data["timestamps"]
                else:
                    render_bkgd = data["color_bkgd"].to(device)
                    rays = namedtuple_map(lambda r: r.to(device), data["rays"])
                    pixels = data["pixels"].to(device)
                    timestamps = data["timestamps"].to(device)

                # rendering
                # rgb, acc, depth, n_rendering_samples, extra = render_image(
                #     radiance_field,
                #     estimator,
                #     rays,
                #     # rendering options
                #     near_plane=near_plane,
                #     render_step_size=render_step_size,
                #     render_bkgd=render_bkgd,
                #     cone_angle=cone_angle,
                #     alpha_thre=alpha_thre,
                #     timestamps=timestamps,
                # )
                rgb, acc, depth, _ = render_image_test(
                    1024,
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                    timestamps=timestamps,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
                ms_ssims = ms_ssim(
                    pixels.permute(2, 0, 1)[None], rgb.permute(2,0,1)[None], data_range=1, size_average=True
                )       
                ssims.append(ms_ssims)
                if test_step == 0:
                    imageio.imwrite(
                        "rgb_test.png",
                        (rgb.cpu().numpy() * 255).astype(np.uint8),
                    )
                    imageio.imwrite(
                        "depth_test.png",
                        depth2img(depth),
                    )
                    imageio.imwrite(
                        "rgb_error.png",
                        (
                            (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                        ).astype(np.uint8),
                    )

        progress_bar.close()
        psnr_avg = sum(psnrs) / len(psnrs)
        ssim_avg = sum(ssims) / len(ssims)
        print(f"evaluation: psnr_avg={psnr_avg}, ssim_avg={ssim_avg}")


if args.render_video:
    if args.load_model:
        checkpoint = torch.load("model.pth")
        radiance_field.load_state_dict(checkpoint["radiance_field"])
        estimator.load_state_dict(checkpoint["occupancy_grid"])
        radiance_field.eval()
        estimator.eval()
    # render video
    with torch.no_grad():
        rgb_frames = []
        depth_frames = []
        progress_bar = tqdm.tqdm(total=len(test_dataset.render_poses), desc=f'rendering video: ')
        render_bkgd = torch.zeros(3, device=device)
        for render_step in range(test_dataset.render_poses.shape[0]):
            progress_bar.update()
            data = test_dataset.get_render_poses(render_step)
            rays = namedtuple_map(lambda r: r.to(device), data["rays"])
            timestamps = data["timestamps"].to(device)
            rgb, acc, depth, _ = render_image_test(
                1024,
                radiance_field,
                estimator,
                rays,
                # rendering options
                near_plane=near_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
                timestamps=timestamps,
            )
            rgb_frames.append(np.flip(rgb.cpu().numpy() * 255, axis=1).astype(np.uint8))
            depth_frames.append(np.flip(depth2img(depth), axis=1))
        progress_bar.close()
        imageio.mimwrite('rgb_render.mp4', rgb_frames, fps=20)
        imageio.mimwrite('depth_render.mp4', depth_frames, fps=20)



if args.gui:
    torch.cuda.empty_cache()
    white_bkgd = torch.ones(3, device=device)
    black_bkgd = torch.zeros(3, device=device)
    render_bkgd = (
        white_bkgd if args.scene in DNERF_SYNTHETIC_SCENES else black_bkgd
    )
    gui_args = {
        'K': mark_invisible_K,
        'img_wh': (test_dataset.width, test_dataset.height),
        'train_camtoworlds': train_dataset.camtoworlds.cpu().numpy(),
        'test_camtoworlds': test_dataset.camtoworlds.cpu().numpy(),
        'train_img_lens': train_dataset.images.shape[0],
        'test_img_lens': test_dataset.images.shape[0],
        'radiance_field': radiance_field, 
        'estimator': estimator,
        'near_plane': near_plane,
        'alpha_thre': alpha_thre,
        'cone_angle': cone_angle,
        # 'test_chunk_size': args.test_chunk_size,
        'render_bkgd': render_bkgd,
        'render_step_size': render_step_size,
        'args_aabb': None,
        'reverse_h': args.scene in DYNERF_SCENES, 
    }
    app = GUI(render_kwargs=gui_args, dnerf=True)
    app.render_gui()
