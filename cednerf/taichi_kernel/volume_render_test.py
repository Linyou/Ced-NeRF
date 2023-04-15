import taichi as ti
from taichi.math import vec3

@ti.kernel
def composite_test(
           sigmas: ti.types.ndarray(ndim=2),
             rgbs: ti.types.ndarray(ndim=2),
          t_start: ti.types.ndarray(ndim=2),
            t_end: ti.types.ndarray(ndim=2),
        pack_info: ti.types.ndarray(ndim=2),
    alive_indices: ti.types.ndarray(ndim=1),
      T_threshold: float,
  aplha_threshold: float,
          opacity: ti.types.ndarray(ndim=2),
            depth: ti.types.ndarray(ndim=2),
              rgb: ti.types.ndarray(ndim=2)):
    
    ti.loop_config(block_dim=256)
    for n in alive_indices:
        start_idx = pack_info[n, 0]
        steps = pack_info[n, 1]
        ray_idx = alive_indices[n]
        if steps == 0:
            alive_indices[n] = -1
        else:
            T = 1 - opacity[ray_idx, 0]

            rgb_temp = vec3(0.0)
            depth_temp = 0.0
            opacity_temp = 0.0

            for s in range(steps):
              s_n = start_idx + s
              t1 = t_start[s_n, 0]
              t2 = t_end[s_n, 0]
              delta = t2 - t1
              a = 1.0 - ti.exp(-sigmas[s_n, 0]*delta)

              if a > aplha_threshold:

                w = a * T
                tmid = (t1 + t2) / 2
                rgbs_vec3 = vec3(
                  rgbs[s_n, 0], rgbs[s_n, 1], rgbs[s_n, 2]
                )
                rgb_temp += w * rgbs_vec3
                depth_temp += w * tmid
                opacity_temp += w
                T *= 1.0 - a

                if T <= T_threshold:
                    alive_indices[n] = -1
                    break

            rgb[ray_idx, 0] += rgb_temp[0]
            rgb[ray_idx, 1] += rgb_temp[1]
            rgb[ray_idx, 2] += rgb_temp[2]
            depth[ray_idx, 0] += depth_temp
            opacity[ray_idx, 0] += opacity_temp