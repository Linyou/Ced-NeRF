import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Callable, Optional, Tuple

from nerfacc import render_weight_from_density, accumulate_along_rays, render_transmittance_from_density

def reduce_along_rays(
    ray_indices: Tensor,
    values: Tensor,
    n_rays: Optional[int] = None,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Accumulate and reduce volumetric values along the ray."""
    assert ray_indices.dim() == 1 and values.dim() == 2
    if not values.is_cuda:
        raise NotImplementedError("Only support cuda inputs.")
    if weights is not None:
        assert (
            values.dim() == 2 and values.shape[0] == weights.shape[0]
        ), "Invalid shapes: {} vs {}".format(values.shape, weights.shape)
        src = weights*values
    else:
        src = values

    if ray_indices.numel() == 0:
        assert n_rays is not None
        return torch.zeros((n_rays, src.shape[-1]), device=values.device)

    if n_rays is None:
        n_rays = int(ray_indices.max()) + 1
    # assert n_rays > ray_indices.max()

    ray_indices = ray_indices.int()
    index = ray_indices[:, None].long().expand(-1, src.shape[-1])
    outputs = torch.zeros((n_rays, src.shape[-1]), device=values.device, dtype=src.dtype)
    outputs.scatter_reduce_(0, index, src, reduce="mean")
    return outputs


def render_weight_from_density_prefix(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    prefix_trans: Tensor,
    packed_info: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Render the weights of the rays through the density field."""
    trans, alphas = render_transmittance_from_density(
        t_starts, t_ends, sigmas, packed_info, ray_indices, n_rays, prefix_trans
    )
    weights = trans * alphas
    return weights, trans, alphas

def rendering(
    # ray marching results
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    ray_indices: torch.Tensor,
    n_rays: int,
    # radiance field
    rgb_sigma_fn: Callable,
    # rendering options
    render_bkgd: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`"""

    # Query sigma/alpha and color with gradients
    rgbs, sigma_results = rgb_sigma_fn(t_starts, t_ends, ray_indices)
    sigmas = sigma_results['density'].squeeze(-1)
    assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
        rgbs.shape
    )
    assert (
        sigmas.shape == t_starts.shape
    ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
    # Rendering: compute weights.
    weights, trans, alphas = render_weight_from_density(
        t_starts,
        t_ends,
        sigmas,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    extras = {
        "weights": weights,
        "alphas": alphas,
        "trans": trans,
        "sigmas": sigmas,
        "rgbs": rgbs,
    }


    # move_norm_render = accumulate_along_rays(
    #     weights.detach(), ray_indices, values=move_norm.unsqueeze(-1), n_rays=n_rays
    # )*10

    if "interal_output" in sigma_results:
        interal_output = sigma_results["interal_output"]
        selector = interal_output["selector"]
        if "latent_losses" in interal_output:
            latent_losses = interal_output["latent_losses"]
            extras["latent_losses"] = reduce_along_rays(
                                ray_indices,
                                values=latent_losses,
                                n_rays=n_rays,
                                weights=weights[:, None],
                            )
        if "weight_losses" in interal_output:
            target_weights = trans[:, None]
            # tiny-cuda-nn output float16
            p_weight = interal_output["weight_losses"].float() 
            weight_loss = F.huber_loss(p_weight, target_weights, reduction='none')
            extras["weight_losses"] = reduce_along_rays(
                            ray_indices,
                            values=weight_loss * selector[:, None],
                            n_rays=n_rays,
                            weights=weights[:, None],
                        )
        

    # move_norm_view = move_norm[:, None]
    # w_dim = sigmas.dim()
    # m_dim = move_norm.dim()
    # assert w_dim == m_dim, f"sigmas: {w_dim} and move :{m_dim} not equal!"

    # with torch.no_grad():
    #     render_move = render_weight_from_density(
    #         t_starts,
    #         t_ends,
    #         move_norm,
    #         ray_indices=ray_indices,
    #         n_rays=n_rays
    #     )
    #     final_move = accumulate_along_rays(
    #         render_move, ray_indices, values=None, n_rays=n_rays
    #     )

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths, extras