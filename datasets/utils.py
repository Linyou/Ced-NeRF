"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import torch
import collections
import numpy as np

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))

Rays_d = collections.namedtuple("Rays", ("origins", "viewdirs", "directions"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))


def ray_to_device(rays_device, device):

    rays = Rays(origins=rays_device.origins.to(device), viewdirs=rays_device.viewdirs.to(device))


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    return np.stack([-vec0, vec1, vec2, pos], axis=1)

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses: np.ndarray) -> np.ndarray:
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)
    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg

def generate_spiral_path(poses: np.ndarray,
                         near_fars: np.ndarray,
                         n_frames=120,
                         n_rots=2,
                         zrate=.5,
                         dt=0.75,
                         percentile=70) -> np.ndarray:
    """Calculates a forward facing spiral path for rendering.

    From https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py
    and https://github.com/apchenstu/TensoRF/blob/main/dataLoader/llff.py

    :param poses: [N, 3, 4]
    :param near_fars:
    :param n_frames:
    :param n_rots:
    :param zrate:
    :param dt:
    :return:
    """
    # center pose
    c2w = average_poses(poses)  # [3, 4]

    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = near_fars.min() * 1.0, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), percentile, 0)
    radii = np.concatenate([radii, [1.]])
    # radii *= 0.5

    # Generate poses for spiral path.
    render_poses = []
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
        position = c2w @ t
        lookat = c2w @ np.array([0, 0, -focal, 1.0])
        z_axis = normalize(position - lookat)
        render_poses.append(viewmatrix(z_axis, up, position))
    return np.stack(render_poses, axis=0)

def generate_hemispherical_orbit(poses: torch.Tensor, n_frames=120):
    """Calculates a render path which orbits around the z-axis.
    Based on https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py
    """
    origins = poses[:, :3, 3]
    radius = torch.sqrt(torch.mean(torch.sum(origins ** 2, dim=-1)))

    # Assume that z-axis points up towards approximate camera hemisphere
    sin_phi = torch.mean(origins[:, 2], dim=0) / radius
    cos_phi = torch.sqrt(1 - sin_phi ** 2)
    render_poses = []

    up = torch.tensor([0., 0., 1.])
    for theta in np.linspace(0., 2. * np.pi, n_frames, endpoint=False):
        camorigin = radius * torch.tensor(
            [cos_phi * np.cos(theta), cos_phi * np.sin(theta), sin_phi])
        render_poses.append(torch.from_numpy(viewmatrix(camorigin, up, camorigin)))

    render_poses = torch.stack(render_poses, dim=0)
    return render_poses