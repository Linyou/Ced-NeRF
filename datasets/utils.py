"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))

Rays_d = collections.namedtuple("Rays", ("origins", "viewdirs", "directions"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))


def ray_to_device(rays_device, device):

    rays = Rays(origins=rays_device.origins.to(device), viewdirs=rays_device.viewdirs.to(device))
