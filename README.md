# SSR-VFD: Spatial Super-Resolution for Vector Field Data Analysis and Visualization
Pytorch implementation for SSR-VFD: Spatial Super-Resolution for Vector Field Data Analysis and Visualization.

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

## Citation 
```
@inproceedings{guo2020ssr,
  title={SSR-VFD: Spatial super-resolution for vector field data analysis and visualization},
  author={Guo, Li and Ye, Shaojie and Han, Jun and Zheng, Hao and Gao, Han and Chen, Danny Z and Wang, Jian-Xun and Wang, Chaoli},
  booktitle={Proceedings of IEEE Pacific isualization Symposium},
  pages={71-80},
  year={2020}}
```
