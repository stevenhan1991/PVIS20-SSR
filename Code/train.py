import argparse
import os
import time
import json
import logging
import random
random.seed(0)
from math import log10
import torch
import torch.utils.data as Data
import torch.nn as nn

from database import TrainDataset
from model import Model, CNN, Discriminator, SRCNN


def train(data_loader, model, MSE_loss, COS_loss, BCE_loss, cos_ratio, optimizer, mask_required, use_cuda, dev, discriminator, optimizer_dis, dis_iter, adv_ratio):
    epoch_mse_loss = 0
    epoch_cos_loss = 0
    epoch_adv_loss = 0
    epoch_real_loss = 0
    epoch_fake_loss = 0
    start_time = time.process_time()
    for step, data in enumerate(data_loader):
        if mask_required:
            lr, hr, mask = data
        else:
            lr, hr = data

        ones = torch.ones((hr.shape[0] * hr.shape[2] * hr.shape[3] * hr.shape[4]))

        if use_cuda:
            lr = lr.cuda()
            hr = hr.cuda()
            if mask_required:
                mask = mask.cuda()
            ones = ones.cuda()

        if dev == False:  # training
            # train discriminator
            if discriminator is not None:
                '''
                for p in model.parameters():
                    p.requires_grad = False
                for p in discriminator.parameters():
                    p.requires_grad = True  
                '''

                for i in range(dis_iter):
                    label_real = torch.rand(hr.shape[0]) * 0.5 + 0.7
                    if use_cuda:
                        label_real = label_real.cuda()
                    output_real = discriminator(hr)
                    real_loss = BCE_loss(output_real, label_real)
                    epoch_real_loss += real_loss.item()

                    res = model(lr)
                    if mask_required:
                        res = res.mul(mask)
                    label_fake = torch.rand(lr.shape[0]) * 0.3
                    if use_cuda:
                        label_fake = label_fake.cuda()
                    output_fake = discriminator(res)
                    fake_loss = BCE_loss(output_fake, label_fake)
                    epoch_fake_loss += fake_loss.item()

                    loss = 0.5 * (real_loss + fake_loss)

                    optimizer_dis.zero_grad()
                    loss.backward()
                    optimizer_dis.step()     
                '''
                for p in model.parameters():
                    p.requires_grad = True
                for p in discriminator.parameters():
                    p.requires_grad = False
                '''

            # train generator
            res = model(lr) # N * 3 * 4x * 4y * 4z
            if mask_required:
                res = res.mul(mask)

            #assert res[torch.isnan(res)].shape[0] == 0

            if discriminator is not None:
                label_fake = torch.ones((lr.shape[0], ), dtype=torch.float)
                if use_cuda:
                    label_fake = label_fake.cuda()
                output_fake = discriminator(res)
                adv_loss = BCE_loss(output_fake, label_fake)

            mse_loss = MSE_loss(res, hr)
            res = res.reshape(res.shape[0], 3, -1).transpose(1, 2).reshape(-1, 3)
            hr = hr.reshape(hr.shape[0], 3, -1).transpose(1, 2).reshape(-1, 3)
            cos_loss = COS_loss(res, hr, ones)
            
            loss = (1 - cos_ratio) * mse_loss + cos_ratio * cos_loss
            if discriminator is not None:
                loss = loss + adv_ratio * adv_loss
                epoch_adv_loss += adv_loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_cos_loss += cos_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                res = model(lr)
                if mask_required:
                    res = res.mul(mask)

                assert res[torch.isnan(res)].shape[0] == 0

                mse_loss = MSE_loss(res, hr)
                res = res.reshape(res.shape[0], 3, -1).transpose(1, 2).reshape(-1, 3)
                hr = hr.reshape(hr.shape[0], 3, -1).transpose(1, 2).reshape(-1, 3)
                cos_loss = COS_loss(res, hr, ones)
                
                epoch_mse_loss += mse_loss.item()
                epoch_cos_loss += cos_loss.item()
    epoch_mse_loss /= (step + 1)
    epoch_cos_loss /= (step + 1)
    epoch_adv_loss /= (step + 1)
    epoch_real_loss /= (step + 1)
    epoch_fake_loss /= (step + 1)
    elapsed_time = time.process_time() - start_time

    if dev == False:
        if discriminator is not None:
            logging.info("train: mse loss: {:.7f} cos loss: {:.6f} dis loss: {:.6f} real loss: {:.6f} fake loss: {:.6f} time: {:.2f}s".format(epoch_mse_loss, epoch_cos_loss, epoch_adv_loss, epoch_real_loss, epoch_fake_loss, elapsed_time))
        else:
            logging.info("train: mse loss: {:.7f} cos loss: {:.6f} time: {:.2f}s".format(epoch_mse_loss, epoch_cos_loss, elapsed_time))
    else:
        logging.info(" dev : mse loss: {:.7f} cos loss: {:.6f}".format(epoch_mse_loss, epoch_cos_loss))

def set_logging(log_file):
    #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M', filename=log_file, filemode='w')
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_file, filemode='a+')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger('').addHandler(console)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-csx", "--cut_size_x", type=int, default=50)
    parser.add_argument("-csy", "--cut_size_y", type=int, default=50)
    parser.add_argument("-csz", "--cut_size_z", type=int, default=50)
    parser.add_argument("-wz", "--weight_z", type=float, default=0.5) # ratio of the dense part
    parser.add_argument("-bs", "--batch_size", type=int, default=1)
    parser.add_argument("-r", "--regularization", type=float, default=0)  # normally 0.0005
    parser.add_argument("-cc", "--cnn_channel", type=int, default=32)
    parser.add_argument("-bn", "--block_number", type=int, default=5)
    parser.add_argument("-pbn", "--post_block_number", type=int, default=0)
    parser.add_argument("-cr", "--cos_ratio", type=float, default=0.001)
    parser.add_argument("-up", "--upscale", type=int, default=4)
    parser.add_argument("-e", "--epoch", type=int, default=500)

    parser.add_argument("-dp", "--data_path", type=str, default="Supernova/")
    parser.add_argument("-ds", "--data_size", type=int, default=123)
    parser.add_argument("-dx", "--data_x", type=int, default=256)
    parser.add_argument("-dy", "--data_y", type=int, default=256)
    parser.add_argument("-dz", "--data_z", type=int, default=256)
    parser.add_argument("-mr", "--mask_required", type=bool, default=False)
    parser.add_argument("-t", "--threshold", type=float, default=1e-2)

    parser.add_argument("-uoc", "--use_one_cnn", type=bool, default=False)

    parser.add_argument("-ug", "--use_gan", type=bool, default=False)
    parser.add_argument("-di", "--dis_iter", type=int, default=1)
    parser.add_argument("-ar", "--adv_ratio", type=float, default=1e-4)

    parser.add_argument("-uc", "--use_cnn", type=bool, default=False)

    parser.add_argument("-p", "--path", type=str, default="result/")
    parser.add_argument("-ce", "--checkpoint_epoch", type=int, default=50)
    parser.add_argument("-ct", "--continue_train", type=bool, default=False)
    args = parser.parse_args()

    if not args.data_path.endswith("/"):
        args.data_path += "/"

    if not args.path.endswith("/"):
        args.path += "/"
    os.makedirs(args.path, exist_ok=True)

    continue_train = args.continue_train
    if continue_train == True: # load previous state and continue training
        with open(args.path + 'config.json') as json_file:
            data = json.load(json_file)
        args = argparse.Namespace(**data)

        i = args.epoch
        while i > 0:
            if os.path.exists(args.path + 'models/model_' + str(i) + '.pk'):
                break
            i -= args.checkpoint_epoch
        last_epoch = i
    else:
        if os.path.exists(args.path + "train.log"):
            os.remove(args.path + "train.log")
        last_epoch = 0

    set_logging(args.path + "train.log")

    # use CUDA to speed up
    use_cuda = torch.cuda.is_available()
    logging.info("CUDA:{}".format(use_cuda))

    # get data
    dataset = TrainDataset(begin=1,
                           end=args.data_size,
                           cut_size_x=args.cut_size_x,
                           cut_size_y=args.cut_size_y,
                           cut_size_z=args.cut_size_z,
                           data_path=args.data_path,
                           data_x=args.data_x,
                           data_y=args.data_y,
                           data_z=args.data_z,
                           weight_z=args.weight_z,
                           mask_required=args.mask_required,
                           threshold=args.threshold,
                           upscale=args.upscale,
                           use_cnn=args.use_cnn)
    size = len(dataset)
    test_size = int(size * 0.2)
    dev_size = int(size * 0.1)
    train_size = size - test_size - dev_size
    index = list(range(size))
    random.shuffle(index)
    train_index = index[ : train_size]
    dev_index = index[train_size : train_size + dev_size]
    test_index = index[train_size + dev_size : train_size + dev_size + test_size]

    train_dataset = Data.Subset(dataset, train_index)
    dev_dataset = Data.Subset(dataset, dev_index)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=False)

    args.train_index = [(i + 1) for i in train_index]
    args.dev_index = [(i + 1) for i in dev_index]
    args.test_index = [(i + 1) for i in test_index]
    args.train_index.sort()
    args.dev_index.sort()
    args.test_index.sort()
    logging.info(json.dumps(vars(args), indent=4))

    if continue_train == False:
        # save configuration
        with open(args.path + 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=4)

    if args.mask_required:
        mask = dataset.get_hr_mask()
        torch.save(mask, args.path + 'mask.pk')

    # initialize model
    if args.use_cnn:
        model = CNN()
    elif not args.use_one_cnn:
        model = Model(channels=args.cnn_channel, block_num=args.block_number, post_block_num=args.post_block_number, upscale=args.upscale)
    else:
        model = SRCNN(channels=args.cnn_channel, block_num=args.block_number, post_block_num=args.post_block_number, upscale=args.upscale, in_channels=3)
    if continue_train == True:
        model.load_state_dict(torch.load(args.path + "models/model_" + str(last_epoch) + ".pk"))
    model.train()
    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.regularization)
    if continue_train:
        optimizer.load_state_dict(torch.load(args.path + "models/optimizer_" + str(last_epoch) + ".pk"))
    MSE_loss = nn.MSELoss()
    COS_loss = nn.CosineEmbeddingLoss()
    BCE_loss = nn.BCELoss()

    discriminator = None
    optimizer_dis = None
    if args.use_gan:
        discriminator = Discriminator(min(args.data_x, args.cut_size_x * args.upscale), min(args.data_y, args.cut_size_y * args.upscale), min(args.data_z, args.cut_size_z * args.upscale))

        if continue_train == True:
            discriminator.load_state_dict(torch.load(args.path + "models/model_dis_" + str(last_epoch) + ".pk"))
           
        discriminator.train()
        if use_cuda:
            discriminator = discriminator.cuda()

        optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
        if continue_train:
            optimizer_dis.load_state_dict(torch.load(args.path + "models/optimizer_dis_" + str(last_epoch) + ".pk"))


    # train model
    path = args.path + "models/"
    os.makedirs(path, exist_ok=True)
    for epoch in range(last_epoch, args.epoch):
        logging.info("Epoch:{:<3}".format(epoch + 1))

        train(train_loader, model, MSE_loss, COS_loss, BCE_loss, args.cos_ratio, optimizer, args.mask_required, use_cuda, False, discriminator, optimizer_dis, args.dis_iter, args.adv_ratio)
        train(dev_loader, model, MSE_loss, COS_loss, BCE_loss, args.cos_ratio, optimizer, args.mask_required, use_cuda, True, discriminator, optimizer_dis, args.dis_iter, args.adv_ratio)

        if (epoch + 1) % args.checkpoint_epoch == 0:
            torch.save(model.state_dict(), path + "model_" + str(epoch + 1) + ".pk")
            torch.save(optimizer.state_dict(), path + "optimizer_" + str(epoch + 1) + ".pk")
            if args.use_gan:
                torch.save(discriminator.state_dict(), path + "model_dis_" + str(epoch + 1) + ".pk")
                torch.save(optimizer_dis.state_dict(), path + "optimizer_dis_" + str(epoch + 1) + ".pk")