import argparse
import os

import numpy as np
from skimage import transform

parser = argparse.ArgumentParser()
# Hyperparameters
parser.add_argument("-up", "--upscale", type=int, default=4)
parser.add_argument("-hrx", "--hrx", type=int, default=128)
parser.add_argument("-hry", "--hry", type=int, default=128)
parser.add_argument("-hrz", "--hrz", type=int, default=128)
parser.add_argument("-sn", "--sample_number", type=int, default=50)
parser.add_argument("-p", "--path", type=str, default="tornado")

args = parser.parse_args()


upscale = args.upscale
hr_x = args.hrx
hr_y = args.hry
hr_z = args.hrz

lr_x = hr_x // upscale
lr_y = hr_y // upscale
lr_z = hr_z // upscale

num = args.sample_number
path = args.path

for i in range(1, num+1):
    print(str(i)+"/"+str(num))

    t = np.fromfile(path+"/high_"+str(i)+".dat", dtype="<f")

    t = t.reshape(-1, 3).transpose()
    t = t.reshape((3, hr_z, hr_y, hr_x))
    x = np.expand_dims(transform.resize(t[0], (lr_z, lr_y, lr_x), order=3), axis=3)
    y = np.expand_dims(transform.resize(t[1], (lr_z, lr_y, lr_x), order=3), axis=3)
    z = np.expand_dims(transform.resize(t[2], (lr_z, lr_y, lr_x), order=3), axis=3)
    t = np.concatenate((x, y, z), axis=3)
    #t = t.reshape((3, -1)).transpose()
    t = t.reshape((-1))
    t.tofile(path+"/low_"+str(upscale)+"_"+str(i)+".dat")

    # x = t[0]
    # y = t[1]
    # z = t[2]

    # x = x.reshape((hr_x, hr_y, hr_z)).transpose()
    # x = transform.resize(x, (lr_x, lr_y, lr_z), order=3).transpose()
    # x = x.reshape((1, -1))

    # y = y.reshape((hr_x, hr_y, hr_z)).transpose()
    # y = transform.resize(y, (lr_x, lr_y, lr_z), order=3).transpose()
    # y = y.reshape((1, -1))

    # z = z.reshape((hr_x, hr_y, hr_z)).transpose()
    # z = transform.resize(z, (lr_x, lr_y, lr_z), order=3).transpose()
    # z = z.reshape((1, -1))

    # res = np.concatenate((x, y, z), axis=0).transpose()
    # res = res.reshape((-1))

    # res.tofile(path+"/low_"+str(upscale)+"_"+str(i)+".dat")
