# Ced-NeRF: A Compact and Efficient Method for Dynamic Neural Radiance Fields

This project is based on Nerfacc.

## Usage

For HyperNeRF dataset:

```unix
python train_real.py --data_root /home/loyot/workspace/Datasets/NeRF/HyberNeRF/  --scene vrig_3dprinter
```

For DyNeRF dataset:

```unix
python train_real.py --data_root /home/loyot/workspace/Datasets/NeRF/3d_vedio_datasets  --scene flame_salmon_1
```

## TODO before release

- [ ] Add DNeRF dataset support 
- [ ] Fuse encoder and MLP after training
- [ ] Support visualization on camera pose and radiance field
- [ ] Add taichi-ngp-renderer support