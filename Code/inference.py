import argparse
import json
import os
import logging
from math import log10, pi
import numpy as np
import torch.utils.data as Data
import torch
import torch.nn as nn

from database import TestDataset
from model import Model, CNN, SRCNN
from train import set_logging


def evaluate(sr, hr, criterion, mask, index_now, use_cuda, path, debug, compute_multiple):
    if mask != None:
        sr = torch.mul(sr, mask) # 3 * 4x * 4y * 4z
    
    cos = torch.sum(sr * hr, dim=0) / (torch.norm(sr, dim=0) * torch.norm(hr, dim=0) + 1e-10)
    cos[cos > 1] = 1
    cos[cos < -1] = -1
    aad = torch.mean(torch.acos(cos)).item() / pi

    loss = criterion(sr.unsqueeze(0), hr.unsqueeze(0)).item()
    L = torch.max(hr).item() - torch.min(hr).item()
    psnr = 10 * log10(L * L / loss)

    if compute_multiple:
        psnr_l = []
        for i in range(3):
            loss = criterion(sr[i].unsqueeze(0), hr[i].unsqueeze(0)).item()
            L = torch.max(hr[i]).item() - torch.min(hr[i]).item()
            psnr_l += [10 * log10(L * L / loss)]
    
    if debug:
        sr = sr.permute(3, 2, 1, 0).reshape(-1)
        if use_cuda:
            sr = sr.cpu()
        sr.numpy().tofile(path + str(index_now) + ".dat")
        if compute_multiple:
            logging.debug("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.5f}".format(index_now, psnr_l[0], psnr_l[1], psnr_l[2], psnr, aad))
        else:
            logging.debug("{}\t{:.4f}\t{:.5f}".format(index_now, psnr, aad))
    return psnr, aad

def inference(data_loader, dataset, model, criterion, use_cuda, mask, path, belong, debug, weight_matrix, compute_multiple):
    epoch_psnr = 0
    epoch_aad = 0
    index_now = -1
    number = 0

    for step, data in enumerate(data_loader):
        lr, hr_index_l, bound_l, file_index_l = data
        if use_cuda:
            lr = lr.cuda()

        with torch.no_grad():
            res = model(lr)
            assert(res[torch.isnan(res)].shape[0] == 0)
        
            for j in range(lr.shape[0]):
                hr_index = hr_index_l[j].item()
                file_index = file_index_l[j].item()
                bound = bound_l[j]

                if index_now != file_index:
                    if index_now != -1:
                        psnr, aad = evaluate(sr / times, hr_whole, criterion, mask, index_now, use_cuda, path, debug, compute_multiple)
                        epoch_psnr += psnr
                        epoch_aad += aad

                    index_now = file_index
                    number += 1
                    sr = torch.zeros((3, dataset.get_x(), dataset.get_y(), dataset.get_z()), dtype=torch.float)
                    times = torch.zeros((3, dataset.get_x(), dataset.get_y(), dataset.get_z()), dtype=torch.float)
                    hr_whole = dataset.get_hr(hr_index)
                    if use_cuda:
                        sr = sr.cuda()
                        times = times.cuda()
                        hr_whole = hr_whole.cuda()
                sr[ : , bound[0].item() : bound[1].item(), bound[2].item() : bound[3].item(), bound[4].item() : bound[5].item()] += res[j].mul(weight_matrix)
                times[ : , bound[0].item() : bound[1].item(), bound[2].item() : bound[3].item(), bound[4].item() : bound[5].item()] += weight_matrix
                assert(not torch.isnan(sr).any())
                assert(not torch.isnan(times).any())
    
    psnr, aad = evaluate(sr / times, hr_whole, criterion, mask, index_now, use_cuda, path, debug, compute_multiple)
    epoch_psnr += psnr
    epoch_aad += aad

    epoch_psnr /= number
    epoch_aad /= number
    logging.info(belong + "\t{:.4f}\t{:.5f}".format(epoch_psnr, epoch_aad))

def get_weight_matrix(use_cuda, x, y, z):
    w = torch.zeros((x, y, z), dtype=torch.float)
    for i in range(x):
        for j in range(y):
            for k in range(z):
                dx = min(i, x - 1 - i)
                dy = min(j, y - 1 - j)
                dz = min(k, z - 1 - k)
                d = min(min(dx, dy), dz) + 1
                w[i, j, k] = d
    if use_cuda:
        w = w.cuda()
    return w


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="result/")
    parser.add_argument("-ca", "--compute_all", type=bool, default=False)
    parser.add_argument("-i", "--interval", type=int, default=50)
    parser.add_argument("-ct", "--compute_train", type=bool, default=False)
    parser.add_argument("-d", "--debug", type=bool, default=False)
    parser.add_argument("-cm", "--compute_multiple", type=bool, default=False)
    args = parser.parse_args()
    compute_all = args.compute_all
    interval = args.interval
    compute_train = args.compute_train
    debug = args.debug
    compute_multiple = args.compute_multiple
    if not args.path.endswith("/"):
        args.path += "/"

    # load configuration
    with open(args.path + 'config.json') as json_file:
        data = json.load(json_file)
    args = argparse.Namespace(**data)

    set_logging(args.path + "inference.log")

    use_cuda = torch.cuda.is_available()
    logging.info("CUDA:{}".format(use_cuda))

    args.test_index.sort()
    test_dataset = TestDataset(range=args.test_index,
                               cut_size_x=args.cut_size_x,
                               cut_size_y=args.cut_size_y,
                               cut_size_z=args.cut_size_z,
                               data_path=args.data_path, 
                               data_x=args.data_x,
                               data_y=args.data_y,
                               data_z=args.data_z,
                               mask_required=args.mask_required,
                               path=args.path,
                               upscale=args.upscale,
                               use_cnn=args.use_cnn)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    if compute_train:
        args.train_index.sort()
        args.dev_index.sort()
        train_dataset = TestDataset(range=args.train_index + args.dev_index,
                                    cut_size_x=args.cut_size_x,
                                    cut_size_y=args.cut_size_y,
                                    cut_size_z=args.cut_size_z,
                                    data_path=args.data_path, 
                                    data_x=args.data_x,
                                    data_y=args.data_y,
                                    data_z=args.data_z,
                                    mask_required=args.mask_required,
                                    path=args.path,
                                    upscale=args.upscale,
                                    use_cnn=args.use_cnn)
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
        
    if args.use_cnn:
        model = CNN()
    elif not args.use_one_cnn:
        model = Model(channels=args.cnn_channel, block_num=args.block_number, post_block_num=args.post_block_number, upscale=args.upscale)
    else:
        model = SRCNN(channels=args.cnn_channel, block_num=args.block_number, post_block_num=args.post_block_number, upscale=args.upscale, in_channels=3)
    criterion = nn.MSELoss()

    mask = None
    if args.mask_required:
        mask = torch.load(args.path + 'mask.pk')

    if not os.path.exists(args.path + 'models/model_' + str(interval) + '.pk'):
        logging.error("No model found!")
        raise SystemExit(0)
    for end_model in reversed(range(interval, args.epoch + 1, interval)):
        if os.path.exists(args.path + 'models/model_' + str(end_model) + '.pk'):
            break
    if compute_all:
        start_model = interval
    else:
        start_model = end_model

    logging.info("Index\tPSNR\tAAD")
    weight_matrix = get_weight_matrix(use_cuda, min(args.data_x, args.cut_size_x * args.upscale), min(args.data_y, args.cut_size_y * args.upscale), min(args.data_z, args.cut_size_z * args.upscale))
    for i in range(start_model, end_model + 1, interval):
        if use_cuda:
            model.load_state_dict(torch.load(args.path + "models/model_" + str(i) + ".pk"))
            model = model.cuda()
            if args.mask_required:
                mask = mask.cuda()
        else:
            model.load_state_dict(torch.load(args.path + "models/model_" + str(i) + ".pk", map_location='cpu'))
        model.eval()

        path = args.path + str(i) + "/"
        if debug:
            os.makedirs(path, exist_ok=True)
        inference(test_loader, test_dataset, model, criterion, use_cuda, mask, path, "Test at " + str(i), debug, weight_matrix, compute_multiple)
        if compute_train:
            inference(train_loader, train_dataset, model, criterion, use_cuda, mask, path, "Train at " + str(i), debug, weight_matrix, compute_multiple)