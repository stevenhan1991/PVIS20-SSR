import numpy as np
import math
import torch
import random
import os
from torch.utils.data.dataset import Dataset


# transform raw data to target format
def transform(nparr, x, y, z):
    vec = torch.reshape(torch.FloatTensor(nparr), (-1, 3)) # xyz * 3
    vec = vec.transpose(0, 1).reshape((3, z, y, x)).permute(0, 3, 2, 1) # 3 * x * y * z
    return vec

# get start and ending point
def rand_range_weight(size, length, weight):
    if size < length:
        index = list(range(0, length - size + 1))
        weights = [(weight ** int(x + size / 2 > length / 2)) * ((1 - weight) ** int(x + size / 2 <= length / 2))  for x in index]
        s = random.choices(index, weights)[0]
        e = s + size
    else:
        s = 0
        e = length
    return s, e

# get start and ending point
def rand_range(size, length):
    if size < length:
        '''
        index = list(range(0, length - size + 1))
        weights = [(4 * math.fabs(x / (length - size) - 0.5) + 1) for x in index]
        s = random.choices(index, weights)[0]
        '''
        s = random.randint(0, length - size)
        e = s + size
    else:
        s = 0
        e = length
    return s, e

# transform HR to its mask
def hr_2_hr_mask(vec, threshold):
    mask = torch.sqrt(torch.sum(torch.mul(vec, vec), dim=0)) # 4x * 4y * 4y
    mask[mask >= threshold] = 1
    mask[mask < threshold] = 0
    return mask

# Customized loading training data
class TrainDataset(Dataset):
    def __init__(self, begin, end, cut_size_x, cut_size_y, cut_size_z, data_path, data_x, data_y, data_z, weight_z, mask_required, threshold, upscale, use_cnn):
        self.lr = []
        self.hr = []
        self.cut_size_x = cut_size_x
        self.cut_size_y = cut_size_y
        self.cut_size_z = cut_size_z
        self.weight_z = weight_z
        self.mask_required = mask_required
        self.hr_mask = None
        if mask_required:
            raw = transform(np.fromfile(data_path + "high_"+ str(begin) + ".dat", dtype='<f'), data_x, data_y, data_z)
            self.hr_mask = hr_2_hr_mask(raw, threshold)

        if use_cnn:
            self.upscale = 1
            self.cut_size_x *= upscale
            self.cut_size_y *= upscale
            self.cut_size_z *= upscale
        else:
            self.upscale = upscale

        for i in range(begin, end + 1):
            if use_cnn:
                self.lr += [transform(np.fromfile(data_path + "high_" + str(upscale) + "_" + str(i) + ".dat", dtype='<f'), data_x, data_y, data_z)]
            else:
                '''
                if upscale == 4:
                    self.lr += [transform(np.fromfile(data_path + "low_" + str(i) + ".dat", dtype='<f'), data_x // self.upscale, data_y // self.upscale, data_z // self.upscale)]
                else:
                '''
                self.lr += [transform(np.fromfile(data_path + "low_" + str(upscale) + "_" + str(i) + ".dat", dtype='<f'), data_x // self.upscale, data_y // self.upscale, data_z // self.upscale)]
            self.hr += [transform(np.fromfile(data_path + "high_"+ str(i) + ".dat", dtype='<f'), data_x, data_y, data_z)]
            if mask_required:
                self.hr[-1] = self.hr[-1].mul(self.hr_mask)

    def __getitem__(self, index):
        x_s, x_e = rand_range(self.cut_size_x, self.lr[0].shape[1])
        y_s, y_e = rand_range(self.cut_size_y, self.lr[0].shape[2])
        z_s, z_e = rand_range_weight(self.cut_size_z, self.lr[0].shape[3], self.weight_z)
        if self.mask_required:
            return self.lr[index][ : , x_s : x_e, y_s : y_e, z_s : z_e], self.hr[index][ : , self.upscale * x_s : self.upscale * x_e, self.upscale * y_s : self.upscale * y_e, self.upscale * z_s : self.upscale * z_e], self.hr_mask[self.upscale * x_s : self.upscale * x_e, self.upscale * y_s : self.upscale * y_e, self.upscale * z_s : self.upscale * z_e]
        else:
            return self.lr[index][ : , x_s : x_e, y_s : y_e, z_s : z_e], self.hr[index][ : , self.upscale * x_s : self.upscale * x_e, self.upscale * y_s : self.upscale * y_e, self.upscale * z_s : self.upscale * z_e]

    def __len__(self):
        return len(self.lr)

    def get_hr_mask(self):
        return self.hr_mask


def save_hr(hr, index, path): # hr: 3 * 4x * 4y * 4z
    hr = hr.permute(3, 2, 1, 0).reshape(-1)
    hr.numpy().tofile(path + str(index) + ".raw")

# Customized loading dev/test data
class TestDataset(Dataset):
    def __init__(self, range, cut_size_x, cut_size_y, cut_size_z, data_path, data_x, data_y, data_z, mask_required, path, upscale, use_cnn):
        self.lr = []
        self.hr_whole = []
        self.hr_index = []
        self.file_index = []
        self.bound = []
        self.x = data_x
        self.y = data_y
        self.z = data_z

        if mask_required:
            mask = torch.load(path + 'mask.pt')
            path = path + "masked_hr/"
            os.makedirs(path, exist_ok=True)

        if use_cnn:
            self.upscale = 1
            cut_size_x *= upscale
            cut_size_y *= upscale
            cut_size_z *= upscale
        else:
            self.upscale = upscale

        for file_index in range:
            if use_cnn:
                whole_lr = transform(np.fromfile(data_path + "high_" + str(upscale) + "_" + str(file_index) + ".dat", dtype='<f'), data_x, data_y, data_z)
            else:
                '''
                if upscale == 4:
                    whole_lr = transform(np.fromfile(data_path + "low_" + str(file_index) + ".dat", dtype='<f'), data_x // self.upscale, data_y // self.upscale, data_z // self.upscale)
                else:
                '''
                whole_lr = transform(np.fromfile(data_path + "low_" + str(upscale) + "_" + str(file_index) + ".dat", dtype='<f'), data_x // self.upscale, data_y // self.upscale, data_z // self.upscale)
            whole_hr = transform(np.fromfile(data_path + "high_" + str(file_index) + ".dat", dtype='<f'), data_x, data_y, data_z)
            if mask_required:
                whole_hr = whole_hr.mul(mask)
                save_hr(whole_hr, file_index, path)
            self.hr_whole += [whole_hr]

            x_s = 0 # start point
            while x_s < whole_lr.shape[1]:
                x_end = False
                x_e = x_s + cut_size_x
                if x_e >= whole_lr.shape[1]: # check if the region has exceed
                    x_e = whole_lr.shape[1]
                    x_s = max(0, x_e - cut_size_x) # move the start point if exceeded
                    x_end = True

                y_s = 0
                while y_s < whole_lr.shape[2]:
                    y_end = False
                    y_e = y_s + cut_size_y
                    if y_e >= whole_lr.shape[2]:
                        y_e = whole_lr.shape[2]
                        y_s = max(0, y_e - cut_size_y)
                        y_end = True
                    
                    z_s = 0
                    while z_s < whole_lr.shape[3]:
                        z_end = False
                        z_e = z_s + cut_size_z
                        if z_e >= whole_lr.shape[3]:
                            z_e = whole_lr.shape[3]
                            z_s = max(0, z_e - cut_size_z)
                            z_end = True

                        self.lr += [whole_lr[ : , x_s : x_e, y_s : y_e, z_s : z_e]]
                        self.file_index += [file_index]
                        self.hr_index += [len(self.hr_whole) - 1]
                        self.bound += [torch.IntTensor([self.upscale * x_s, self.upscale * x_e, self.upscale * y_s, self.upscale * y_e, self.upscale * z_s, self.upscale * z_e])]

                        if z_end:
                            break
                        z_s += cut_size_z // 4 * 3
                    if y_end:
                        break
                    y_s += cut_size_y // 4 * 3
                if x_end:
                    break
                x_s += cut_size_x // 4 * 3
            
    def __getitem__(self, index):
        return self.lr[index], self.hr_index[index], self.bound[index], self.file_index[index]

    def __len__(self):
        return len(self.lr)

    def get_hr(self, index):
        return self.hr_whole[index]

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z