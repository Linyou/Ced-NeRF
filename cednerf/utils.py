import random
import numpy as np

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from typing import Optional

from datasets.utils import Rays, namedtuple_map

from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc import traverse_grids, ray_aabb_intersect

from .render import rendering, render_weight_from_density_prefix, accumulate_along_rays
from .taichi_kernel import composite_test

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def render_image(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            sigmas = radiance_field.query_density(positions, t)
        else:
            sigmas = radiance_field.query_density(positions)
        return sigmas['density'].squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas

    results = []
    extra_info = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = estimator.sampling(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
        extras['ray_indices'] = ray_indices
        extras['t_starts'] = t_starts
        extras['t_ends'] = t_ends
        extra_info.append(extras)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
        extra_info
    )


@torch.no_grad()
def render_image_test(
    max_samples: int,
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    early_stop_eps: float = 1e-4,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape
    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = rays.origins[ray_indices]
        t_dirs = rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas['density'].squeeze(-1)

    N_rays = rays.origins.shape[0]
    device = rays.origins.device
    opacity = torch.zeros(N_rays, 1, device=device)
    depth = torch.zeros(N_rays, 1, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)

    ray_mask_id = torch.arange(N_rays, device=device)

    min_samples = 1 if cone_angle==0 else 4

    iter_samples = total_samples = 0

    rays_o = rays.origins
    rays_d = rays.viewdirs
    
    near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
    far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

    t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, estimator.aabbs)  

    n_grids = estimator.binaries.size(0)

    if n_grids > 1:
        t_sorted, t_indices = torch.sort(torch.cat([t_mins, t_maxs], -1), -1)
    else:
        t_sorted = torch.cat([t_mins, t_maxs], -1)
        t_indices = torch.arange(
            0, n_grids * 2, device=t_mins.device, dtype=torch.int64
        ).expand(N_rays, n_grids * 2)

    while iter_samples < max_samples:

        N_alive = len(ray_mask_id)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        iter_samples += N_samples

        # debug_str = f"samples: {iter_samples}, N_alive: {N_alive}, N_samples: {N_samples} | "

        # ray marching
        (
            intervals, 
            samples, 
            termination_planes
        ) = traverse_grids(
            # rays
            rays_o,  # [n_rays, 3]
            rays_d,  # [n_rays, 3]
            # grids
            estimator.binaries,  # [m, resx, resy, resz]
            estimator.aabbs,  # [m, 6]
            # options
            near_planes,  # [n_rays]
            far_planes,  # [n_rays]
            render_step_size,
            cone_angle,
            N_samples,
            ray_mask_id, 
            # pre-compute intersections
            t_sorted,  # [n_rays, m*2]
            t_indices,  # [n_rays, m*2]
            hits,  # [n_rays, m]
        )
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices_local = intervals.ray_indices[intervals.is_right]
        ray_indices = samples.ray_indices[samples.is_valid]
        packed_info = samples.packed_info

        # get rgb and sigma from radiance field
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
         # volume rendering
        # native cuda scan
        weights, _, _ = render_weight_from_density_prefix(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices_local,
            n_rays=N_alive,
            prefix_trans=1 - opacity[ray_indices].squeeze(-1),
        )
        rgb[ray_mask_id] += accumulate_along_rays(
            weights, values=rgbs, ray_indices=ray_indices_local, n_rays=N_alive
        )
        opacity[ray_mask_id] += accumulate_along_rays(
            weights, values=None, ray_indices=ray_indices_local, n_rays=N_alive
        )
        depth[ray_mask_id] += accumulate_along_rays(
            weights,
            values=(t_starts + t_ends)[..., None] / 2.0,
            ray_indices=ray_indices_local,
            n_rays=N_alive,
        )
        # debug_str += f"t_min[{ray_mask_id[0]}]: {near_planes[ray_mask_id[0]]} | "
        # taichi kernel 
        # composite_test(
        #     # input args
        #     sigmas.contiguous(), 
        #     rgbs.contiguous(), 
        #     t_starts.contiguous(),
        #     t_ends.contiguous(), 
        #     packed_info.contiguous(), 
        #     ray_mask_id.contiguous(), 
        #     early_stop_eps, 
        #     alpha_thre,
        #     # output args
        #     opacity.contiguous(), 
        #     depth.contiguous(), 
        #     rgb.contiguous()
        # )
        # update near_planes using termination planes
        near_planes[ray_mask_id] = termination_planes
        # update rays status
        mask = torch.logical_or(
            opacity[ray_mask_id, 0] > (1 - early_stop_eps) ,
            packed_info[:, 1] == 0
        )
        ray_mask_id = ray_mask_id[~mask]
        total_samples += ray_indices.shape[0]
        # print(debug_str)
    
    rgb = rgb + render_bkgd * (1.0 - opacity)
    depth = depth / opacity.clamp_min(torch.finfo(rgbs.dtype).eps)

    return (
        rgb.view((*rays_shape[:-1], -1)),
        opacity.view((*rays_shape[:-1], -1)),
        depth.view((*rays_shape[:-1], -1)),
        total_samples,
    )