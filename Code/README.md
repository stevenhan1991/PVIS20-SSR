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

## SolarPlume
python transfer.py -up 4 -hrx 252 -hry 252 -hrz 1024 -sn 29 -p ../dataset/SolarPlume

python bicubic.py -p 0.7
python train.py -dp ../dataset/SolarPlume -ds 29 -dx 252 -dy 252 -dz 1024 -csx 32 -csy 32 -csz 32 -up 4 -e 2000 -wz 0.7 -uc True -p solar_cnn_4
python inference.py -p solar_cnn_4 -ca True -i 400
python train.py -dp ../dataset/SolarPlume -ds 29 -dx 252 -dy 252 -dz 1024 -csx 32 -csy 32 -csz 32 -up 4 -e 2000 -wz 0.7 -ug True -p solar_gan_4

python transfer.py -up 2 -hrx 252 -hry 252 -hrz 1024 -sn 29 -p ../dataset/SolarPlume
python train.py -dp ../dataset/SolarPlume -ds 29 -dx 252 -dy 252 -dz 1024 -csx 64 -csy 64 -csz 64 -up 2 -e 2000 -wz 0.7 -p solar_xxx_2
python bicubic.py -p solar_xxx_2