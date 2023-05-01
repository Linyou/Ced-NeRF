import numpy as np
import taichi as ti
import torch
from taichi.math import uvec3, ivec3
from torch.cuda.amp import custom_bwd, custom_fwd

data_type = ti.f32
torch_type = torch.float32

@ti.kernel
def torch2ti(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        field[I] = data[I]


@ti.kernel
def ti2torch(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = field[I]


@ti.kernel
def ti2torch_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        grad[I] = field.grad[I]


@ti.kernel
def torch2ti_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        field.grad[I] = grad[I]

@ti.kernel
def random_initialize(data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = (ti.random() * 2.0 - 1.0) * 1e-4


@ti.kernel
def ti_copy(data1: ti.template(), data2: ti.template()):
    for I in ti.grouped(data1):
        data1[I] = data2[I]


@ti.kernel
def ti_copy_array(data1: ti.types.ndarray(), data2: ti.types.ndarray()):
    for I in ti.grouped(data1):
        data1[I] = data2[I]


@ti.kernel
def ti_copy_field_array(data1: ti.template(), data2: ti.types.ndarray()):
    for I in ti.grouped(data1):
        data1[I] = data2[I]

tf_vec3 = ti.types.vector(n=3, dtype=data_type)
ivec6 = ti.types.vector(n=6, dtype=ti.i32)
feat_dim = 2
levels = 8

@ti.kernel
def _kernel(
        xyzs: ti.template(), 
        plane_embedding: ti.template(),
        plane_scales: ti.template(),
        xyzs_embedding: ti.template(), 
        B: ti.i32,
        plane_res: ti.i32,):

    # get hash table embedding
    # ti.loop_config(block_dim=128)
    for i, sn in ti.ndrange(B, feat_dim*levels):
        j = sn // levels
        level = sn % levels
        xyz = ti.Vector([xyzs[i, 0], xyzs[i, 1], xyzs[i, 2]])
        plane_scale = plane_scales[level]
        xy_plane_feat = 0.
        yz_plane_feat = 0.
        zx_plane_feat = 0.

        # plane query
        pos = xyz * (plane_scale-1) + 0.5
        pos_grid_uint = ti.cast(ti.floor(pos), ti.int32)
        pos -= pos_grid_uint

        pos_fuse = ti.Vector([
            pos[1], pos[2], 
            pos[0], pos[2], 
            pos[0], pos[1]
        ])
        pos_grid_uint_fuse = ti.Vector([
            pos_grid_uint[1], pos_grid_uint[2], 
            pos_grid_uint[0], pos_grid_uint[2], 
            pos_grid_uint[0], pos_grid_uint[1]]
        )

        for idx in ti.static(range(4)):
            w = tf_vec3(1.)
            pos_grid_local = ivec6(0)

            for d in ti.static(range(2)):
                if (idx & (1 << d)) == 0:
                    pos_grid_local[d::2] = pos_grid_uint_fuse[d::2]
                    w *= 1 - pos_fuse[d::2]
                else:
                    pos_grid_local[d::2] = pos_grid_uint_fuse[d::2] + 1
                    w *= pos_fuse[d::2]

            # convert pos_grid_local to high res
            pos_grid_local = ti.cast(pos_grid_local / plane_scale * plane_res, ti.i32)

            index = ivec3(0)
            stride = 1
            for i in ti.static(range(2)):
                index += pos_grid_local[i::2] * stride
                stride *= plane_res

            xy_index_table = index[0] * feat_dim
            yz_index_table = plane_res**2*feat_dim + index[1] * feat_dim
            zx_index_table = plane_res**2*feat_dim*2 + index[2] * feat_dim

            xy_plane_feat += w[0] * plane_embedding[xy_index_table + j]
            yz_plane_feat += w[1] * plane_embedding[yz_index_table + j]
            zx_plane_feat += w[2] * plane_embedding[zx_index_table + j]

        sum_feat = xy_plane_feat * yz_plane_feat * zx_plane_feat

        level_offset = level*(feat_dim*4)
        xyzs_embedding[i, j + level_offset] = xy_plane_feat 
        xyzs_embedding[i, j + feat_dim + level_offset] = yz_plane_feat 
        xyzs_embedding[i, j + feat_dim*2 + level_offset] = zx_plane_feat 
        xyzs_embedding[i, j + feat_dim*3 + level_offset] = sum_feat



class TriPlaneEncoder(torch.nn.Module):

    def __init__(
            self,
            plane_res=4096,
            batch_size=8192,
            data_type=data_type
        ):
        super(TriPlaneEncoder, self).__init__()

        self.plane_res = plane_res
        self.n_output_dims = feat_dim * levels * 4

        self.plane_scales = ti.field(dtype=data_type,shape=(levels, ))
        b = np.exp(np.log(self.plane_res / 16) / (levels - 1))
        for i in range(levels):
            self.plane_scales[i] = int(
                np.ceil(
                    16 * np.exp(i * np.log(b)) - 1.0
                )
            ) + 1
        self.plane_feat = feat_dim

        self.plane_embedding = torch.nn.Parameter(
            torch.zeros(
                (self.plane_res**2) * 3 * self.plane_feat, 
                dtype=torch_type
            ),
            requires_grad=True
        )
                        
        random_initialize(self.plane_embedding)

        self.plane_embedding_fields = ti.field(
            data_type, 
            shape=(self.plane_res**2 * 3 * self.plane_feat, ), 
            needs_grad=True
        )
        self.output_fields = ti.field(
            dtype=data_type,
            shape=(batch_size * 1024, self.n_output_dims),
            needs_grad=True
        )
        self.input_fields = ti.field(
            dtype=data_type,
            shape=(batch_size * 1024, 3),
            needs_grad=True
        )
        
        self._encode_kernel = _kernel

        self.register_buffer(
            'plane_grad', 
            torch.zeros(
                self.plane_res**2 * 3 * self.plane_feat, 
                dtype=torch_type
            ),
            persistent=False,
        )
        self.register_buffer(
            'output_embedding',
            torch.zeros(batch_size * 1024, self.n_output_dims, dtype=torch_type),
            persistent=False,
        )

        class _module_function(torch.autograd.Function):

            @staticmethod
            @custom_fwd(cast_inputs=torch_type)
            def forward(ctx, input_pos, plane_params):
                output_embedding = self.output_embedding[:input_pos.
                                                         shape[0]].contiguous(
                                                         )
                torch2ti(self.input_fields, input_pos.contiguous())
                torch2ti(self.plane_embedding_fields, plane_params.contiguous())

                self._encode_kernel(
                    self.input_fields,
                    self.plane_embedding_fields,
                    self.plane_scales,
                    self.output_fields,
                    input_pos.shape[0],
                    self.plane_res,
                )
                ti2torch(self.output_fields, output_embedding)

                return output_embedding

            @staticmethod
            @custom_bwd
            def backward(ctx, doutput):

                self.zero_grad()

                torch2ti_grad(self.output_fields, doutput.contiguous())
                self._encode_kernel.grad(
                    self.input_fields,
                    self.plane_embedding_fields,
                    self.plane_scales,
                    self.output_fields,
                    doutput.shape[0],
                    self.plane_res,
                )
                ti2torch_grad(self.plane_embedding_fields,
                                   self.plane_grad.contiguous())
                return None, self.plane_grad

        self._module_function = _module_function

    def zero_grad(self):
        self.plane_embedding_fields.grad.fill(0.)

    def forward(self, positions):
        return self._module_function.apply(positions, self.plane_embedding)
