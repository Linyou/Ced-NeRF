import nerfvis
import numpy as np


class NerfvisCallback():
    def __init__(self, vis_blocking=True):
        self.vis_blocking = vis_blocking
        self.vis = nerfvis.Scene("nerf")

    def display(self, port=8889):
        self.vis.display(port=port, serve_nonblocking=(not self.vis_blocking))

    def render_nerf(self, aabb, eval_fn):
        # occ_non_vis = occupancy_grid.get_non_visiable()
        # for level, occ_i in enumerate(occ_non_vis):
        #     occ_i = occ_i.cpu().numpy()
        #     print(occ_i.shape)
        #     vis.add_points(
        #         f"occ_{level}",
        #         occ_i,
        #         point_size=2**(5 - level),  
        #     )

        # def nerfvis_eval_fn(x, dirs):
        #     t = torch.zeros(*x.shape[:-1], 1, device=x.device)
        #     density, embedding = radiance_field.query_density(
        #         x, t, return_feat=True
        #     )
        #     embedding = embedding.expand(-1, dirs.shape[1], -1)
        #     dirs = dirs.expand(embedding.shape[0], -1, -1)
        #     rgb = radiance_field._query_rgb(
        #         dirs, embedding=embedding, apply_act=False
        #     )
        #     return rgb, density

        self.vis.remove("nerf")
        self.vis.add_nerf(
            name="nerf",
            eval_fn=eval_fn,
            center=((aabb[3:] + aabb[:3]) / 2.0).tolist(),
            radius=((aabb[3:] - aabb[:3]) / 2.0).max().item(),
            use_dirs=True,
            reso=128,
            sigma_thresh=1.0,
        )
        self.vis.display(port=8889, serve_nonblocking=True)

    def add_camera_frustum(
            self, 
            name, 
            focal_length, 
            image_width, 
            image_height, 
            z, r, t,
        ):
        self.vis.add_camera_frustum(
            name,
            focal_length=focal_length,
            image_width=image_width,
            image_height=image_height,
            z=z,
            r=r,
            t=t,
        )

    def add_boxes(self, aabb, aabb_bkgd):
        p1 = aabb[:3]
        p2 = aabb[3:]
        verts, segs = [
            [p1[0], p1[1], p1[2]],
            [p1[0], p1[1], p2[2]],
            [p1[0], p2[1], p2[2]],
            [p1[0], p2[1], p1[2]],
            [p2[0], p1[1], p1[2]],
            [p2[0], p1[1], p2[2]],
            [p2[0], p2[1], p2[2]],
            [p2[0], p2[1], p1[2]],
        ], [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        self.vis.add_lines(
            "aabb",
            np.array(verts).astype(dtype=np.float32),
            segs=np.array(segs),
        )

        p1 = aabb_bkgd[:3]
        p2 = aabb_bkgd[3:]
        verts, segs = [
            [p1[0], p1[1], p1[2]],
            [p1[0], p1[1], p2[2]],
            [p1[0], p2[1], p2[2]],
            [p1[0], p2[1], p1[2]],
            [p2[0], p1[1], p1[2]],
            [p2[0], p1[1], p2[2]],
            [p2[0], p2[1], p2[2]],
            [p2[0], p2[1], p1[2]],
        ], [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        self.vis.add_lines(
            "aabb_bkgd",
            np.array(verts).astype(dtype=np.float32),
            segs=np.array(segs),
        )