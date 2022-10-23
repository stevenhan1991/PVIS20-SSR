# SSR-VFD: Spatial Super-resolution for Vector Field Data Analysis and Visualization
Pytorch implementation for SSR-VFD: Spatial Super-resolution for Vector Field Data Analysis and Visualization.

## Vessel
```
python train.py -dp dataset/Vessel -ds 50 -dx 280 -dy 260 -dz 180 -mr True -t 1e-2 -p vessel
python inference.py -p vessel
```

## Tornado
```
python train.py -dp dataset/Tornado -ds 48 -dx 128 -dy 128 -dz 128 -p tornado
python inference.py -p tornado
```

## Supernova
```bash
python train.py -dp Supernova -ds 123 -dx 256 -dy 256 -dz 256 -p supernova
python inference.py -p supernova
```

If you want to change the upscale
```bash
python3 train.py -dp Supernova_2 -ds 123 -dx 256 -dy 256 -dz 256 -p supernova -up 2
```
## Cylinder
```
python transfer.py -up 4 -hrx 192 -hry 64 -hrz 48 -sn 100 -p ../dataset/Cylinder
python train.py -dp ../dataset/Cylinder -ds 100 -dx 192 -dy 64 -dz 48 -csx 50 -csy 50 -csz 50 -up 4 -p cylinder_xxx_4
python inference.py -p cylinder_xxx_4 -ct True
python bicubic.py -p cylinder_xxx_4

python train.py -dp ../dataset/Cylinder -ds 100 -dx 192 -dy 64 -dz 48 -csx 50 -csy 50 -csz 50 -up 4 -uc True -p cylinder_cnn_4
python inference.py -p cylinder_cnn_4 -ct True

python train.py -dp ../dataset/Cylinder -ds 100 -dx 192 -dy 64 -dz 48 -up 4 -ug True -p cylinder_gan_4
python inference.py -p cylinder_gan_4 -ct True

python transfer.py -up 2 -hrx 192 -hry 64 -hrz 48 -sn 100 -p ../dataset/Cylinder
python train.py -dp ../dataset/Cylinder -ds 100 -dx 192 -dy 64 -dz 48 -csx 100 -csy 100 -csz 100 -up 2 -p cylinder_xxx_2
python inference.py -p cylinder_xxx_2 -ct True
python bicubic.py -p cylinder_xxx_2

python transfer.py -up 8 -hrx 192 -hry 64 -hrz 48 -sn 100 -p ../dataset/Cylinder
python train.py -dp ../dataset/Cylinder -ds 100 -dx 192 -dy 64 -dz 48 -csx 25 -csy 25 -csz 25 -up 8 -p cylinder_xxx_8
python inference.py -p cylinder_xxx_8 -ct True
python bicubic.py -p cylinder_xxx_8
```

## continue training
python train.py -dp ../dataset/Tornado -ds 48 -dx 128 -dy 128 -dz 128 -csx 50 -csy 50 -csz 50 -up 4 -p 5
python train.py -p 5 -ct True
python inference.py -p 5 -ct True -ca True

## Citation 
```
@inproceedings{guo2020ssr,
  title={SSR-VFD: Spatial super-resolution for vector field data analysis and visualization},
  author={Guo, Li and Ye, Shaojie and Han, Jun and Zheng, Hao and Gao, Han and Chen, Danny Z and Wang, Jian-Xun and Wang, Chaoli},
  booktitle={Proceedings of IEEE Pacific isualization Symposium},
  pages={71-80},
  year={2020}
}

```
