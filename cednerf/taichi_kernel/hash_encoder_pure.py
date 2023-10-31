import torch
import taichi as ti
import numpy as np

from taichi.math import uvec3, vec3
from torch.cuda.amp import custom_bwd, custom_fwd
ti.init(arch=ti.cuda, half2_vectorization=True)

data_type = ti.f16

def align_to(x, y):
    return int((x+y-1)/y)*y

def res_in_level_np(
        level_i, 
        base_res, 
        log_per_level_scale
    ):
    result = np.ceil(
        float(base_res) * np.exp(
            float(level_i) * log_per_level_scale
        ) - 1.0
    )
    return float(result + 1)

def scale_in_level_np(
        base_res, 
        max_res,
        levels,
    ):
    result = np.log(
        float(max_res) / float(base_res)
    ) / float(levels - 1)
    return result

def build_hash_encoder_kernel(
    log_per_level_scale,
    base_res: float = 16.0,
    hash_level: int = 16,
    feat_dim: int = 2,
    begin_fast_hash_level: int = 16,
):
    """
    This function constructs a Taichi kernel that encodes
    3D coordinates into a hash map with multiple levels of resolution.

    Args:
    base_res (float, optional): Base resolution of the hash map. Default is 16.
    hash_level (int, optional): Number of levels in the hash map. Default is 16.
    feature_per_level (int, optional): Number of features per level. Default is 2.
    begin_fast_hash_level (int, optional): The level at which the fast hash method
    starts. Default is 16.

    Returns:
    A Taichi kernel, hash_encoder_kernel.
    """

    # Type
    feat_vec = ti.types.vector(
        n=feat_dim, 
        dtype=data_type,
    )

    @ti.func
    def grid_scale(level, log_scale, base_res):
        exp_scale = ti.exp(level * log_scale)
        return base_res * exp_scale - 1.0

    @ti.kernel
    def hash_encoder_kernel(
            xyzs: ti.types.ndarray(vec3), 
            table: ti.types.ndarray(feat_vec),
            output_embedding: ti.types.ndarray(feat_vec), 
            hash_map_sizes: ti.types.ndarray(), 
            offsets: ti.types.ndarray(), 
            B: ti.i32,
        ):
        # get hash table embedding
        ti.loop_config(block_dim=hash_level)
        for i, level in ti.ndrange(B, hash_level):
            xyz = xyzs[i]
            offset = offsets[level]
            map_size = hash_map_sizes[level]

            primes = uvec3(ti.uint32(1), ti.uint32(2654435761), ti.uint32(805459861)) * t

            scale = grid_scale(level, log_per_level_scale, base_res)

            pos = xyz * scale + 0.5
            pos_grid = ti.cast(ti.floor(pos), ti.uint32)
            pos -= ti.cast(pos_grid, ti.f32)


            a = pos[0]
            b = pos[1]
            c = pos[2]

            pos_x = pos_grid[0]
            pos_y = pos_grid[1]
            pos_z = pos_grid[2]


            pos_000 = (((pos_x) * primes[0]) ^ ((pos_y) * primes[1]) ^ ((pos_z) * primes[2])) % map_size
            pos_001 = (((pos_x) * primes[0]) ^ ((pos_y) * primes[1]) ^ ((pos_z + 1) * primes[2])) % map_size
            pos_010 = (((pos_x) * primes[0]) ^ ((pos_y + 1) * primes[1]) ^ ((pos_z) * primes[2])) % map_size
            pos_011 = (((pos_x) * primes[0]) ^ ((pos_y + 1) * primes[1]) ^ ((pos_z + 1) * primes[2])) % map_size
            pos_100 = (((pos_x + 1) * primes[0]) ^ ((pos_y) * primes[1]) ^ ((pos_z) * primes[2])) % map_size
            pos_101 = (((pos_x + 1) * primes[0]) ^ ((pos_y) * primes[1]) ^ ((pos_z + 1) * primes[2])) % map_size
            pos_110 = (((pos_x + 1) * primes[0]) ^ ((pos_y + 1) * primes[1]) ^ ((pos_z) * primes[2])) % map_size
            pos_111 = (((pos_x + 1) * primes[0]) ^ ((pos_y + 1) * primes[1]) ^ ((pos_z + 1) * primes[2])) % map_size

            w000 = (1.0 - a) * (1.0 - b) * (1.0 - c)
            w001 = (1.0 - a) * (1.0 - b) * c
            w010 = (1.0 - a) * b * (1.0 - c)
            w011 = (1.0 - a) * b * c
            w100 = a * (1.0 - b) * (1.0 - c)
            w101 = a * (1.0 - b) * c
            w110 = a * b * (1.0 - c)
            w111 = a * b * c

            output_embedding[i, level] = ti.cast(
                w000 * table[(offset+pos_000)] + \
                w001 * table[(offset+pos_001)] + \
                w010 * table[(offset+pos_010)] + \
                w011 * table[(offset+pos_011)] + \
                w100 * table[(offset+pos_100)] + \
                w101 * table[(offset+pos_101)] + \
                w110 * table[(offset+pos_110)] + \
                w111 * table[(offset+pos_111)],
                data_type
            )

    # TODO: implement xyzs gradient backward 
    @ti.kernel
    def hash_encoder_backward_kernel(
            xyzs: ti.types.ndarray(vec3), 
            hash_map_sizes: ti.types.ndarray(), 
            offsets: ti.types.ndarray(), 
            output_grad: ti.types.ndarray(feat_vec),
            hash_grad: ti.types.ndarray(feat_vec),
            B: ti.i32,
        ):
        # get hash table embedding
        ti.loop_config(block_dim=hash_level)
        for i, level in ti.ndrange(B, hash_level):
            xyz = xyzs[i]
            offset = offsets[level]
            map_size = hash_map_sizes[level]
            dL_dy = output_grad[i, level]

            primes = uvec3(ti.uint32(1), ti.uint32(2654435761), ti.uint32(805459861)) * t

            scale = grid_scale(level, log_per_level_scale, base_res)

            pos = xyz * scale + 0.5
            pos_grid = ti.cast(ti.floor(pos), ti.uint32)
            pos -= ti.cast(pos_grid, ti.f32)

            a = pos[0]
            b = pos[1]
            c = pos[2]

            pos_x = pos_grid[0]
            pos_y = pos_grid[1]
            pos_z = pos_grid[2]

            w000 = (1.0 - a) * (1.0 - b) * (1.0 - c)
            w001 = (1.0 - a) * (1.0 - b) * c
            w010 = (1.0 - a) * b * (1.0 - c)
            w011 = (1.0 - a) * b * c
            w100 = a * (1.0 - b) * (1.0 - c)
            w101 = a * (1.0 - b) * c
            w110 = a * b * (1.0 - c)
            w111 = a * b * c

            pos_000 = (((pos_x) * primes[0]) ^ ((pos_y) * primes[1]) ^ ((pos_z) * primes[2])) % map_size
            pos_001 = (((pos_x) * primes[0]) ^ ((pos_y) * primes[1]) ^ ((pos_z + 1) * primes[2])) % map_size
            pos_010 = (((pos_x) * primes[0]) ^ ((pos_y + 1) * primes[1]) ^ ((pos_z) * primes[2])) % map_size
            pos_011 = (((pos_x) * primes[0]) ^ ((pos_y + 1) * primes[1]) ^ ((pos_z + 1) * primes[2])) % map_size
            pos_100 = (((pos_x + 1) * primes[0]) ^ ((pos_y) * primes[1]) ^ ((pos_z) * primes[2])) % map_size
            pos_101 = (((pos_x + 1) * primes[0]) ^ ((pos_y) * primes[1]) ^ ((pos_z + 1) * primes[2])) % map_size
            pos_110 = (((pos_x + 1) * primes[0]) ^ ((pos_y + 1) * primes[1]) ^ ((pos_z) * primes[2])) % map_size
            pos_111 = (((pos_x + 1) * primes[0]) ^ ((pos_y + 1) * primes[1]) ^ ((pos_z + 1) * primes[2])) % map_size


            ws = ti.types.vector(8, ti.f16)(
                w000, w001, w010, w011, w100, w101, w110, w111
            )
            index_all = ti.types.vector(8, ti.u32)(
                pos_000, pos_001, pos_010, pos_011, 
                pos_100, pos_101, pos_110, pos_111
            ) + offset

            if dL_dy.any():
                for d in ti.static(range(8)):
                    cur_w = dL_dy * ws[d]
                    if cur_w.any():
                        hash_grad[index_all[d]] += cur_w


    return hash_encoder_kernel, hash_encoder_backward_kernel

class HashEncoder(torch.nn.Module):

    def __init__(
        self,
        max_params: float=2**19,
        levels: int=16.0,
        base_res: float=16.0,
        max_res: float=2048.0,
        feature_per_level: int=2,  
    ):
        super(HashEncoder, self).__init__()

        # b=1.3195079565048218 fix value for 16 -> 1024
        self.log_b = scale_in_level_np(
            base_res=base_res,
            max_res=max_res,
            levels=levels,
        )
        # self.log_b = 1.587401032447815
        self.base_res = base_res
        self.hash_level = levels
        self.max_params = max_params
        self.feature_per_level = feature_per_level
        self.out_dim = feature_per_level * levels

        self.register_buffer(
            'offsets',
            torch.zeros(levels, dtype=torch.int32),
            persistent=False
        )
        self.register_buffer(
            'hash_map_sizes',
            torch.zeros(levels, dtype=torch.int32),
            persistent=False
        )

        offset = 0
        begin_fast_hash_level = levels
        for i in range(levels):
            resolution = res_in_level_np(
                i, base_res, self.log_b
            )
            full_size = resolution**3
            # Ensure that the parameter size is a multiple of 8.
            full_size_aligned = align_to(full_size, 8)

            # Restricted the parameter size using max_params.
            params_size_i = min(max_params, full_size_aligned)
            # print("resolution: ", resolution)

            self.offsets[i] = offset
            self.hash_map_sizes[i] = params_size_i

            # Record the first level that begins to use fast_hash
            if full_size > params_size_i:
                if begin_fast_hash_level == levels:
                    begin_fast_hash_level = i
            
            offset += params_size_i

        self.begin_fast_hash_level = begin_fast_hash_level
        self.total_param_size = offset * feature_per_level

        print(
            f'Hash Encoder: '
            f'base_res={base_res} '
            f'max_res={max_res} '
            f'hash_level={levels} '
            f'feat_per_level={feature_per_level} '
            f'per_level_scale={self.log_b} '
            f'total_hash_size={offset} '
        )

        self.hash_table = torch.nn.Parameter(
            torch.zeros(
                self.total_param_size//feature_per_level,
                feature_per_level,
                dtype=torch.float32,
            ),
            requires_grad=True
        )
        torch.nn.init.uniform_(self.hash_table, -1e-4, 1e-4)
        self.register_buffer(
            'hash_grad',
            torch.zeros_like(
                self.hash_table, 
                dtype=torch.float32
            ).reshape(-1, feature_per_level),
        )

        (
            self._hash_encoder_kernel, 
            self._hash_encoder_backward_kernel 
        ) = build_hash_encoder_kernel(
            self.log_b,
            base_res=self.base_res,
            hash_level=self.hash_level,
            feat_dim=self.feature_per_level,
            begin_fast_hash_level=self.begin_fast_hash_level,
        )
        # self.grad_scaler = 2.0
        

        # TODO: use a method to build the autograd function
        class _module_function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_pos, params):
                # If no output gradient is provided, no need to
                # automatically materialize it as torch.zeros.
                # ctx.set_materialize_grads(False)

                output_embedding = torch.empty(
                    input_pos.shape[0], 
                    self.hash_level, 
                    self.feature_per_level,
                    dtype=torch.float16,
                    device=input_pos.device, 
                )
                self._hash_encoder_kernel(
                    input_pos,
                    params,
                    output_embedding,
                    self.hash_map_sizes,
                    self.offsets,
                    input_pos.shape[0],
                )
                ctx.save_for_backward(
                    input_pos
                )

                return output_embedding

            @staticmethod
            def backward(ctx, doutput):
                input_pos, = ctx.saved_tensors
                hash_grad = (
                    self.hash_grad.zero_().contiguous()
                )
                self._hash_encoder_backward_kernel(
                    input_pos,
                    self.hash_map_sizes,
                    self.offsets,
                    doutput,
                    hash_grad,
                    input_pos.shape[0],
                )
                return None, hash_grad
        self._module_function = _module_function.apply
        
    def forward(self, positions):
        return self._module_function(
            positions.contiguous(), 
            self.hash_table.to(torch.float16).contiguous(),
        ).view(-1, self.out_dim)
    