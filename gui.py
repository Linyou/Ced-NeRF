import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import cv2

import time

import numpy as np
import torch
import torch.nn.functional as F

from datasets.utils import Rays
from cednerf.utils import render_image_test

import taichi as ti

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).cpu().numpy().astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img.astype(np.float32)

@ti.kernel
def write_buffer(W:ti.i32, H:ti.i32, x: ti.types.ndarray(), final_pixel:ti.template()):
    for i, j in ti.ndrange(W, H):
        for p in ti.static(range(3)):
            final_pixel[i, j][p] = x[j, i, p]

import warnings; warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_rays(K, pose, width, height, opengl=True):

    x, y = torch.meshgrid(
        torch.arange(width, device=device, dtype=torch.float16),
        torch.arange(height, device=device, dtype=torch.float16),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    # generate rays
    c2w = pose[None, ...]  # (num_rays, 3, 4)
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - K[0, 2] + 0.5) / K[0, 0],
                (y - K[1, 2] + 0.5) / K[1, 1]
                * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [num_rays, 3]

    # [n_cams, height, width, 3]
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    # print(c2w[:, :3, -1])
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )

    # origins = torch.reshape(origins, (height, width, 3)).to(torch.float16)
    # viewdirs = torch.reshape(viewdirs, (height, width, 3)).to(torch.float16)
    # directions = torch.reshape(directions, (height, width, 3)).to(torch.float16)
    origins = torch.reshape(origins, (height, width, 3))
    viewdirs = torch.reshape(viewdirs, (height, width, 3))
    directions = torch.reshape(directions, (height, width, 3))

    rays = Rays(origins=origins, viewdirs=viewdirs)

    return rays


class Camera:
    """
    Camera class from: https://github.com/kwea123/ngp_pl/blob/master/show_gui.py
    """
    def __init__(self, K, img_wh, pose, r, center=None):
        self.K = K
        self.W, self.H = img_wh
        self.radius = r
        pose_np = pose
        if center is not None:
            self.center = center
        else:
            self.center = np.zeros(3)
        self.rot = np.eye(3)
        # self.center = pose_np[20][:3, 3]
        # self.rot = pose_np[50][:3, :3]
        self.res_defalut = pose_np[20]
        self.rotate_speed = 0.8

        self.inner_rot = np.eye(3)

    def reset(self, pose=None, aabb=None):
        self.rot = np.eye(3)
        self.inner_rot = np.eye(3)
        self.center = np.zeros(3)
        self.radius = 1.5
        if pose is not None:
            self.rot = pose[:3, :3]

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4)
        res[2, 3] += self.radius
        # rotate
        rot = np.eye(4)
        rot[:3, :3] = self.rot
        res = rot @ res
        # inner rotate
        rot = np.eye(4)
        rot[:3, :3] = self.inner_rot
        res = res @ rot
        # translate
        res[:3, 3] += self.center
        # return res

        # print("res_defalut: ", self.res_defalut)
        # print("res: ", res)
        # return self.res_defalut
        return res

    def orbit(self, dx, dy):
        rotvec_x = self.rot[:, 1] * np.radians(-100*self.rotate_speed * dx)
        rotvec_y = self.rot[:, 0] * np.radians(-100*self.rotate_speed * dy)
        self.rot = R.from_rotvec(rotvec_y).as_matrix() @ \
                   R.from_rotvec(rotvec_x).as_matrix() @ \
                   self.rot

    def inner_orbit(self, dx, dy):
        rotvec_x = self.inner_rot[:, 1] * np.radians(-100*self.rotate_speed * dx)
        rotvec_y = self.inner_rot[:, 0] * np.radians(-100*self.rotate_speed * dy)
        self.inner_rot = R.from_rotvec(rotvec_y).as_matrix() @ \
                         R.from_rotvec(rotvec_x).as_matrix() @ \
                         self.inner_rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 1e-4 * self.rot @ np.array([dx, dy, dz])


class GUI:
    def __init__(self, radius=1.5, render_kwargs=None, dnerf=False, opengl=True):

        device = "cuda:0"

        K = render_kwargs['K']
        img_wh = render_kwargs['img_wh']
        self.train_camtoworlds = render_kwargs['train_camtoworlds']
        self.test_camtoworlds = render_kwargs['test_camtoworlds']
        self.train_img_lens = render_kwargs['train_img_lens']
        self.test_img_lens = render_kwargs['test_img_lens']


        self.radiance_field = render_kwargs['radiance_field']
        self.estimator = render_kwargs['estimator']

        self.near_plane = render_kwargs['near_plane']
        self.render_step_size = render_kwargs['render_step_size']
        self.alpha_thre = render_kwargs['alpha_thre']

        self.cone_angle = render_kwargs['cone_angle']
        self.render_bkgd = render_kwargs['render_bkgd']
        self.args_aabb = render_kwargs['args_aabb']

        self.get_rays_func = get_rays


        self.radiance_field.eval()
        self.estimator.eval()


        self.cam = Camera(K, img_wh, self.test_camtoworlds, r=radius)
        self.W, self.H = img_wh

        # placeholders
        self.dt = 0
        self.mean_samples = 0
        self.img_mode = 0

        if dnerf:
            self.timestamps = torch.tensor([0.0], device=device)
        else:
            self.timestamps = None
        self.max_samples = 100

        self.opengl = opengl

    @torch.no_grad()
    def render_frame(self):
        t = time.time()
        # print(cam.pose)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            rays = self.get_rays_func(self.cam.K, torch.cuda.FloatTensor(self.cam.pose), self.W, self.H, opengl=self.opengl)
            rgb, _, depth, n_rendering_samples = render_image_test(
                self.max_samples,
                self.radiance_field,
                self.estimator,
                rays,
                # rendering options
                near_plane=self.near_plane,
                render_step_size=self.render_step_size,
                render_bkgd=self.render_bkgd,
                cone_angle=self.cone_angle,
                alpha_thre=self.alpha_thre,
                # dnerf
                timestamps=self.timestamps,
            )

            depth = depth.squeeze(-1)
            self.dt = time.time()-t
            self.mean_samples = n_rendering_samples/(self.W * self.H)

            if self.img_mode == 0:
                return rgb
            elif self.img_mode == 1:
                return depth2img(depth)/255.0



    def render_gui(self):

        ti.init(arch=ti.cuda, offline_cache=True)

        W, H = self.W, self.H
        print("W:", type(W))
        final_pixel = ti.Vector.field(3, dtype=float, shape=(W, H))

        window = ti.ui.Window('Window Title', (W, H),)
        canvas = window.get_canvas()
        gui = window.get_gui()


        # GUI controls variables
        last_orbit_x = None
        last_orbit_y = None

        last_inner_x = None
        last_inner_y = None

        timestamps = 0.0
        last_timestamps = 0.0

        playing = False

        test_view = 0
        train_view = 0
        last_train_view = 0
        last_test_view = 0

        ref_c2w = self.train_camtoworlds[train_view]

        train_views_size = self.train_img_lens-1
        test_views_size = self.test_img_lens-1

        self.radiance_field.eval()
        while window.running:

            if window.is_pressed(ti.ui.RMB):
                curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
                if last_orbit_x is None or last_orbit_y is None:
                    last_orbit_x, last_orbit_y = curr_mouse_x, curr_mouse_y
                else:
                    dx = curr_mouse_x - last_orbit_x
                    dy = curr_mouse_y - last_orbit_y
                    self.cam.orbit(dx, -dy)
                    last_orbit_x, last_orbit_y = curr_mouse_x, curr_mouse_y

            elif window.is_pressed(ti.ui.MMB):
                curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
                if last_inner_x is None or last_inner_y is None:
                    last_inner_x, last_inner_y = curr_mouse_x, curr_mouse_y
                else:
                    dx = curr_mouse_x - last_inner_x
                    dy = curr_mouse_y - last_inner_y
                    self.cam.inner_orbit(dx, -dy)
                    last_inner_x, last_inner_y = curr_mouse_x, curr_mouse_y
            else:
                last_orbit_x = None
                last_orbit_y = None

                last_inner_x = None
                last_inner_y = None

            if window.is_pressed('w'):
                self.cam.scale(0.2)
            if window.is_pressed('s'):
                self.cam.scale(-0.2)
            if window.is_pressed('a'):
                self.cam.pan(-500, 0.)
            if window.is_pressed('d'):
                self.cam.pan(500, 0.)
            if window.is_pressed('e'):
                self.cam.pan(0., -500)
            if window.is_pressed('q'):
                self.cam.pan(0., 500)

            with gui.sub_window("Options", 0.01, 0.01, 0.4, 0.3) as w:
                self.cam.rotate_speed = w.slider_float('rotate speed', self.cam.rotate_speed, 0.1, 1.)

                timestamps = w.slider_float('timestamps', timestamps, 0., 1.)
                if last_timestamps != timestamps:
                    last_timestamps = timestamps
                    self.timestamps[0] = timestamps

                if gui.button('play'):
                    playing = True
                if gui.button('pause'):
                    playing = False

                if playing:
                    timestamps += 0.01
                    if timestamps > 1.0:
                        timestamps = 0.0

                self.img_mode = w.checkbox("show depth", self.img_mode)

                train_view = w.slider_int('train view', train_view, 0, train_views_size)
                test_view = w.slider_int('test view', test_view, 0, test_views_size)

                if last_train_view != train_view:
                    last_train_view = train_view
                    ref_c2w = self.train_camtoworlds[train_view]
                    self.cam.reset(
                        self.train_camtoworlds[train_view],
                        aabb=self.args_aabb
                    )

                if last_test_view != test_view:
                    last_test_view = test_view
                    self.cam.reset(
                        self.test_camtoworlds[test_view],
                        aabb=self.args_aabb
                    )

                cam_pose = self.cam.pose
                w.text(f'samples per rays: {self.mean_samples} s/r')
                w.text(f'render times: {1000*self.dt:.2f} ms')
                w.text(f'radius: {self.cam.radius}')
                w.text(f'pose:')
                w.text(f'{self.cam.rot[0]}')
                w.text(f'{self.cam.rot[1]}')
                w.text(f'{self.cam.rot[2]}')
                w.text(f'c2w:')
                w.text(f'{cam_pose[0]}')
                w.text(f'{cam_pose[1]}')
                w.text(f'{cam_pose[2]}')
                w.text(f'{cam_pose[2]}')
                w.text(f'ref c2w:')
                w.text(f'{ref_c2w[0]}')
                w.text(f'{ref_c2w[1]}')
                w.text(f'{ref_c2w[2]}')
                w.text(f'{ref_c2w[2]}')

            render_buffer = self.render_frame()
            write_buffer(W, H, render_buffer, final_pixel)
            canvas.set_image(final_pixel)
            window.show()


