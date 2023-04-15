from torch import Tensor
from torch_efficient_distloss import flatten_eff_distloss

def distortion(
    ray_ids: Tensor, weights: Tensor, t_starts: Tensor, t_ends: Tensor
) -> Tensor:

    interval = t_ends - t_starts
    tmid = (t_starts + t_ends) / 2

    return flatten_eff_distloss(weights.squeeze(-1), tmid.squeeze(-1), interval.squeeze(-1), ray_ids)
