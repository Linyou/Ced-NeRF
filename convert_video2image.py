import numpy as np
import os
import glob
from tqdm import tqdm
import ffmpeg
from PIL import Image
import json
from multiprocessing import Process

import re
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


data_path_root = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/'
# scenes = ['coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_salmon_1', 'flame_steak', 'sear_steak']
# scenes = ["coffee_martini", ]
scenes = ['cook_spinach', 'cut_roasted_beef', 'flame_salmon_1', 'flame_steak', 'sear_steak']
ori_res = (2028, 2704)
dst_res = (int(2704/2), int(2028/2))

def exc_fn(data_path, scene, video_list, idx):
    video_list.sort(key=natural_keys)
    list_json = {}
    with tqdm(position=idx) as progress:
        videos_collect = []
        for video_path in video_list:
            # print("start processing video:", video_path)
            out, _ = (
                ffmpeg
                .input(video_path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel="quiet")
                .global_args('-hide_banner')
                .run(capture_stdout=True)
            )

            video = (
                np
                .frombuffer(out, np.uint8)
                .reshape([-1, 2028, 2704, 3])
            )
            
            # cam name
            basename = os.path.basename(video_path).split('.')[0]
            root = os.path.join(data_path, "images_split")
            os.makedirs(root, exist_ok=True)
            video_len = video.shape[0]
            # print("start saving images")
            # with tqdm(total=video.shape[0], position=idx+1, desc=f'{idx}-{basename}') as progress:
            images_collect = []
            progress.set_description_str(f'{scene}-{basename}')
            progress.reset(total=video_len)
            for idx in range(video_len):
                progress.update()
                save_dir = os.path.join(root, f'{idx}', 'input')
                os.makedirs(save_dir, exist_ok=True)
                img0 = Image.fromarray(video[idx]).resize(dst_res)
                img0_path = os.path.join(save_dir, f'{basename}.png')
                img0.save(img0_path)
                images_collect.append({
                    "path": img0_path, 
                    'idx': idx, 
                    'weight': dst_res[0], 
                    'height': dst_res[1]
                })
            progress.refresh()
            
            videos_collect.append({
                "images": images_collect,
                "video_path": video_path,
                "video_name": basename,
            })

            del out
            del video

    list_json = {'videos': videos_collect, 'scene': scene, 'data_path': data_path}
    with open(os.path.join(data_path,'images_split_list.json'), 'w') as outfile:
        json.dump(list_json, outfile, indent=4)


if __name__ == '__main__':
    p_list = []
    for idx, scene in enumerate(scenes):
        data_path = os.path.join(data_path_root, scene)
        video_list = glob.glob(os.path.join(data_path, '*.mp4'))
        # exc_fn(video_list)
        p = Process(target=exc_fn, args=(data_path, scene, video_list, idx))
        p_list.append(p)
        p.start()

    for p in p_list:
        p.join()


