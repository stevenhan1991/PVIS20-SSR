import logging
import argparse
import json
import os
from skimage import transform
import numpy as np
import torch
import torch.nn as nn
from math import log10
import statistics

from train import set_logging
from inference import evaluate


def run(range, args, belong):
    psnr_l = []
    aad_l = []

    for i in range:
        lr = np.fromfile(args.data_path + "low_" + str(args.upscale) + "_" + str(i) + ".dat", dtype="<f").reshape(-1, 3).transpose().reshape((3, args.data_z // args.upscale, args.data_y // args.upscale, args.data_x // args.upscale))
        hr = np.fromfile(args.data_path + "high_" + str(i) + ".dat", dtype="<f").reshape(-1, 3).transpose().reshape((3, args.data_z, args.data_y, args.data_x))

        sr_x = np.expand_dims(transform.resize(lr[0], (args.data_z, args.data_y, args.data_x), order=3), axis=3)
        sr_y = np.expand_dims(transform.resize(lr[1], (args.data_z, args.data_y, args.data_x), order=3), axis=3)
        sr_z = np.expand_dims(transform.resize(lr[2], (args.data_z, args.data_y, args.data_x), order=3), axis=3)

        sr = np.concatenate((sr_x, sr_y, sr_z), axis=3) # 4z * 4y * 4x * 3
        use_cuda = torch.cuda.is_available()

        hr = torch.tensor(hr)
        sr = torch.tensor(sr)
        if use_cuda:
            hr = hr.cuda()
            sr = sr.cuda()
        hr = hr.permute(0, 3, 2, 1)
        sr = sr.permute(3, 2, 1, 0) # 3 * 4x * 4y * 4z
        
        criterion = nn.MSELoss()
        mask = None
        if args.mask_required:
            mask = torch.load(args.path + 'mask.pk')

        psnr, aad = evaluate(sr, hr, criterion, mask, i, use_cuda, args.data_path + "high_" + str(args.upscale) + "_", True)
        psnr_l += [psnr]
        aad_l += [aad]

    logging.info(belong + " avg\t{:.4f}\t{:.5f}".format(statistics.mean(psnr_l), statistics.mean(aad_l)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="result/")
    args = parser.parse_args()
    if not args.path.endswith("/"):
        args.path += "/"

    # load configuration
    with open(args.path + 'config.json') as json_file:
        data = json.load(json_file)
    args = argparse.Namespace(**data)

    set_logging(args.data_path + "bicubic_" + str(args.upscale) + ".log")

    logging.info("Index\tPSNR\tAAD")
    args.test_index.sort()
    run(args.test_index, args, "Test")
    args.train_index.sort()
    args.dev_index.sort()
    run(args.train_index + args.dev_index, args, "Train")