"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import math
from typing import Callable, List, Union
import torch
import torch.nn.functional as F

from .utils import trunc_exp
from .encoder import SinusoidalEncoderWithExp, SinusoidalEncoder
from .taichi_kernel.triplane import TriPlaneEncoder

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


MOVING_STEP = 1/(4096*1)
class DNGPradianceField(torch.nn.Module):
    """Instance-NGP radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        geo_feat_dim: int = 15,
        base_resolution: int = 16,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        dst_resolution: int = 4096,
        log2_hashmap_size: int = 19,
        use_feat_predict: bool = False,
        use_weight_predict: bool = False,
        moving_step: float = MOVING_STEP,
        use_div_offsets: bool = False,
        use_time_embedding: bool = False,
        use_time_attenuation: bool = False,
        time_inject_before_sigma: bool = True,
        hash4motion: bool = False,    
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation

        self.box_scale = (aabb[3:] - aabb[:3]).max() / 2
        print("self.box_scale: ", self.box_scale)

        self.geo_feat_dim = geo_feat_dim
        per_level_scale = math.exp(
            math.log(dst_resolution * 1 / base_resolution) / (n_levels - 1)
        )  # 1.4472692012786865
        # per_level_scale = 1.4472692012786865

        print('--DNGPradianceField configuration--')
        print(f'  moving_step: {moving_step}')
        print(f'  hash b: {per_level_scale:6f}')
        print(f'  use_div_offsets: {use_div_offsets}')
        print(f'  use_feat_predict: {use_feat_predict}')
        print(f'  use_weight_predict: {use_weight_predict}')
        print(f'  use_time_embedding: {use_time_embedding}')
        print(f'  use_time_attenuation: {use_time_attenuation}')
        print(f'  time_inject_before_sigma: {time_inject_before_sigma}')
        print('-----------------------------------')

        self.use_feat_predict = use_feat_predict
        self.use_weight_predict = use_weight_predict
        self.use_time_embedding = use_time_embedding
        self.use_time_attenuation = use_time_attenuation
        self.MOVING_STEP = moving_step
        self.use_div_offsets = use_div_offsets

        self.time_inject_before_sigma = time_inject_before_sigma

        self.loose_move = False

        self.motion_input_dim = 3 + 1
        self.motion_output_dim = 3 * 2 if use_div_offsets else 3

        self.return_extra = False

        if hash4motion:
            # hash table for time encoding
            self.xyz_wrap = tcnn.NetworkWithInputEncoding(
                n_input_dims=self.motion_input_dim,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 4,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": 8,
                    "per_level_scale":  math.exp(
                        math.log(64 / 8) / (4 - 1)
                    )  # 1.4472692012786865,
                },
                n_output_dims=self.motion_output_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
        else:
            self.xyz_wrap = tcnn.NetworkWithInputEncoding(
                n_input_dims=self.motion_input_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 4,
                            "otype": "Frequency",
                            "n_frequencies": 4
                        },
                        # {"otype": "Identity", "n_bins": 4, "degree": 4},
                    ],
                },
                n_output_dims=self.motion_output_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )


        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                        # {"otype": "Identity", "n_bins": 4, "degree": 4},
                    ],
                },
            )



        self.hash_encoder = tcnn.Encoding(
            n_input_dims=num_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )
        # self.hash_encoder = TriPlaneEncoder()

        input_dim4base = self.hash_encoder.n_output_dims
        self.geo_feat_dim_head = self.geo_feat_dim

        if self.use_time_embedding:
            self.time_encoder = SinusoidalEncoder(1, 0, 4, True)
            self.time_encoder_feat = SinusoidalEncoderWithExp(1, 0, 6, True)

            if self.use_time_attenuation:
                if self.time_inject_before_sigma:
                    input_dim4base += self.time_encoder_feat.latent_dim
                else:
                    self.geo_feat_dim_head  += self.time_encoder_feat.latent_dim
            else:
                if self.time_inject_before_sigma:
                    input_dim4base += self.time_encoder.latent_dim
                else:
                    self.geo_feat_dim_head  += self.time_encoder.latent_dim

        self.mlp_base = tcnn.Network(
            n_input_dims=input_dim4base,
            n_output_dims=1+self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        if self.geo_feat_dim > 0:
            self.mlp_head = tcnn.Network(
                n_input_dims=(
                    (
                        self.direction_encoding.n_output_dims
                        if self.use_viewdirs
                        else 0
                    )
                    + self.geo_feat_dim_head 
                ),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )


        if self.use_feat_predict:
            self.mlp_feat_prediction = tcnn.NetworkWithInputEncoding(
                n_input_dims=num_dim+1,
                encoding_config={
                    "otype": "Frequency",
                    "n_frequencies": 3
                },
                n_output_dims=self.hash_encoder.n_output_dims,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )

        if self.use_weight_predict:
            self.mlp_weight_prediction = tcnn.NetworkWithInputEncoding(
                n_input_dims=num_dim+1,
                encoding_config={
                    "otype": "Frequency",
                    "n_frequencies": 3
                },
                n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )

    def swap_fused_sigmas(self):
        '''
            This function is to replace the separated 
            hash_encoder and the mlp_base to a fused one.
            It should be much faster.
        '''
        pass

    def query_move(self, x, t):
        offsets = self.xyz_wrap(torch.cat([x, t], dim=-1))
        if self.use_div_offsets:
            grid_move = offsets[:, 0:3]*self.MOVING_STEP
            fine_move = (torch.special.expit(offsets[:, 3:])*2 - 1)*self.MOVING_STEP
        else:
            grid_move = offsets*self.MOVING_STEP
            fine_move = 0

        move = grid_move + fine_move

        return x + move, move

    def query_density(self, x, t, return_feat: bool = False, return_interal: bool = False):

        if (not self.loose_move) and x.shape[0] > 0:
            x_move, move = self.query_move(
                x.view(-1, self.num_dim), 
                t.view(-1, 1)
            )
        else:
            x_move = x.view(-1, self.num_dim)
            move = torch.zeros_like(x_move[:, :1])

        aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
        x_move = (x_move - aabb_min) / (aabb_max - aabb_min)

        x = x_move.view_as(x)

        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        hash_feat = self.hash_encoder(x_move)

        if self.use_time_embedding:
            with torch.no_grad():
                if self.use_time_attenuation:
                    move = torch.linalg.norm(move.detach(), dim=-1)
                    # print("max move: ", move.max())
                    # print("min move: ", move.min())
                    time_encode = self.time_encoder_feat(
                        t.view(-1, 1), move.view(-1, 1)
                    )
                else:
                    time_encode = self.time_encoder(t.view(-1, 1))

            if self.time_inject_before_sigma:
                cat_feat = torch.cat([hash_feat, time_encode], dim=-1)
            else:
                cat_feat = hash_feat
        else:
            cat_feat = hash_feat


        x = (
            self.mlp_base(cat_feat)
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )

        results = {}
        results['density'] = density

        if return_feat:
            if self.use_time_embedding and (not self.time_inject_before_sigma):
                results['base_mlp_out'] = torch.cat([base_mlp_out, time_encode], dim=-1)
            else:
                results['base_mlp_out'] = base_mlp_out

        if return_interal:
            if self.use_feat_predict or self.use_weight_predict:
                temp_feat = torch.cat([x_move, t], dim=-1)
                interal_output = {}
                interal_output['selector'] = selector

                if self.use_feat_predict:
                    predict_feat = self.mlp_feat_prediction(temp_feat)
                    loss_feat = F.huber_loss(predict_feat, hash_feat, reduction='none') * selector[..., None]
                    interal_output['latent_losses'] = loss_feat

                if self.use_weight_predict:
                    interal_output['weight_losses'] = self.mlp_weight_prediction(temp_feat)

                results['interal_output'] = interal_output

        return results

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = dir / torch.linalg.norm(
                dir, dim=-1, keepdims=True
            )

            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim_head)], dim=-1)
        else:
            h = embedding.reshape(-1, self.geo_feat_dim_head)
        rgb = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        if apply_act:
            rgb = torch.sigmoid(rgb)
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        t: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"

        sigma_results = self.query_density(
            positions, 
            t, 
            return_feat=True,
            return_interal=self.training
        )
        embedding = sigma_results['base_mlp_out']
        rgb = self._query_rgb(directions, embedding=embedding)

        return rgb, sigma_results
