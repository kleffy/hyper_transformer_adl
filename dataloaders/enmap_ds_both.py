# from osgeo import gdal
import collections
import os

import numpy as np
import pandas as pd
import torch
from torch.utils import data
import torchvision.transforms as transforms
import lmdb

import matplotlib.pyplot as plt
from scipy.signal import convolve

class enmap_dataset(data.Dataset):
    def __init__(
        self, config, 
        keys, 
        is_train=True, is_dhp=False, want_DHP_MS_HR=False
    ):
        self.keys = keys
        self.env = lmdb.open(
                            os.path.join(config["enmap_dataset"]["lmdb_save_dir"],
                                         config["enmap_dataset"]["lmdb_file_name"]), 
                            readonly=True, lock=False)
        self.channels = config["enmap_dataset"]["spectral_bands"]
        self.lmdb_save_dir = config["enmap_dataset"]["lmdb_save_dir"]
        self.lmdb_file_name = config["enmap_dataset"]["lmdb_file_name"]
        self.csv_file_name = config["enmap_dataset"]["csv_file_name"]
        self.pred_dir = config["enmap_dataset"]["pred_dir"]
        
        
        self.split  = "train" if is_train else "val"        #Define train and validation splits
        self.config = config                                #Configuration file
        self.want_DHP_MS_HR = want_DHP_MS_HR                #This ask: DO we need DIP up-sampled output as dataloader output?
        self.is_dhp         = is_dhp                        #This checks: "Is this DIP training?"
        self.dir = self.config["enmap_dataset"]["data_dir"] #Path to enmap Center dataset 
    
        self.augmentation = self.config["enmap_dataset"]["augmentation"]   #Augmentation needed or not? 

        self.LR_crop_size = (self.config["enmap_dataset"]["LR_size"], self.config["enmap_dataset"]["LR_size"])  #Size of the the LR-HSI

        self.HR_crop_size = [self.config["enmap_dataset"]["HR_size"], self.config["enmap_dataset"]["HR_size"]]  #Size of the HR-HSI

        # # cv2.setNumThreads(0)    # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing

        self.files = collections.defaultdict(list)
        for f in self.keys:
            # self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": os.path.join(self.pred_dir, f),
                }
            )

    def __len__(self):
        return len(self.keys)
    
    def _augmentaion(self, MS_image, PAN_image, reference):
        N_augs = 4
        aug_idx = torch.randint(0, N_augs, (1,))
        if aug_idx==0:
            #Horizontal Flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
        elif aug_idx==1:
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])
        elif aug_idx==2:
            #Horizontal flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])

        return MS_image, PAN_image, reference

    def getHSIdata(self, index):
        key = self.keys[index]
        with self.env.begin() as txn:
            data = txn.get(key.encode())
        
        key_split = key.split("_")
        
        # dtype = np.uint16 if key_split[-9] == 'uint16' else np.int16

        data = np.frombuffer(data, dtype=np.float32)

        data_size = (int(key_split[-2]), int(key_split[-1]))
        channels = int(key_split[-4])
        data = data.reshape((channels, data_size[0], data_size[1]))
        
        image_dict = self.files[self.split][index]
        data_dtype = data.dtype
        # print(data)
        #data = self.normalize_hyperspectral_image(data)
        
        if self.config["enmap_dataset"]["normalize"]:
            data = self.mean_normalization(data, data_dtype)
        
        # print(f'after : {data}')
        # reference = self._extract_percentile_range(data, 1, 99)
        reference = data
        PAN_image = np.mean(reference[0:50, :, :], axis=0)
        # breakpoint()
        ratio       = 4
        sig         = (1/(2*(2.7725887)/ratio**2))**0.5
        MS_image = self.downsample(ref=reference, ratio=ratio, kernel_size=(9, 9), sig=sig, start_pos=(0, 0))
        # plt.imshow(MS_image[5,:,:], cmap='gray')
        # plt.show()
            
        # COnvert inputs into torch tensors #.transpose(1, 2, 0)
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0))
        PAN_image   = torch.from_numpy(np.array(PAN_image)/1.0)
        reference   = torch.from_numpy((np.array(reference)/1.0)) 
        
        if not self.config["enmap_dataset"]["normalize"]:
        # Max Normalization
            MS_image    = MS_image/self.config["enmap_dataset"]["max_value"]
            PAN_image   = PAN_image/self.config["enmap_dataset"]["max_value"]
            reference   = reference/self.config["enmap_dataset"]["max_value"]           

        #If split = "train" and augment = "true" do augmentation
        if self.split == "train" and self.augmentation:
            MS_image, PAN_image, reference = self._augmentaion(MS_image, PAN_image, reference)

        # if self.split == "train" and index == len(self.files[self.split]) - 1:
        #     np.random.shuffle(self.files[self.split])

        return image_dict, MS_image, PAN_image, reference

    def mean_normalization(self, data, data_dtype):
        if data_dtype == 'uint16':
            mean = np.load(self.config["enmap_dataset"]["l1_mean_file"])
            std = np.load(self.config["enmap_dataset"]["l1_std_file"])
            normalize_transform = transforms.Normalize(mean=mean, std=std + 1e-8)
            data = normalize_transform(torch.from_numpy(np.float32(data)))
        else:
            mean = np.load(self.config["enmap_dataset"]["l2_mean_file"])
            std = np.load(self.config["enmap_dataset"]["l2_std_file"])
            normalize_transform = transforms.Normalize(mean=mean, std=std + 1e-8)
            data = normalize_transform(torch.from_numpy(np.float32(data)))
        return data.numpy()

    
    def __getitem__(self, index):

        image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)

        return image_dict, MS_image, PAN_image, reference
    
    def downsample(self, ref, ratio, kernel_size, sig, start_pos):
    # Create the Gaussian kernel
        kernel = np.zeros(kernel_size)
        center = [i // 2 for i in kernel_size]
        for x in range(kernel_size[0]):
            for y in range(kernel_size[1]):
                kernel[x, y] = np.exp(-0.5 * ((x - center[0]) ** 2 + (y - center[1]) ** 2) / sig ** 2)
        kernel /= np.sum(kernel)

        # Apply the kernel to the reference image
        ref = ref.astype(np.float32)
        output = np.zeros((ref.shape[0], ref.shape[1] // ratio, ref.shape[2] // ratio))
        for c in range(ref.shape[0]):
            filtered = convolve(ref[c], kernel, mode='same')
            output[c] = filtered[start_pos[0]::ratio, start_pos[1]::ratio]

        return output

    def _extract_percentile_range(self, data, lo, hi):
        plo = np.percentile(data, lo, axis=(1, 2), keepdims=True)
        phi = np.percentile(data, hi, axis=(1, 2), keepdims=True)
        data = np.clip(data, plo, phi)
        with np.errstate(divide='ignore', invalid='ignore'):
            data = np.where(phi - plo == 0, 0, (data - plo) / (phi - plo))
        return data

    
if __name__ == '__main__':
    import json
    import matplotlib.pyplot as plt
    
    def read_csv_keys(csv_file, csv_file_col_name):
        df = pd.read_csv(csv_file)
        keys = df[csv_file_col_name].tolist()
        return keys
    
    config_file = r"/vol/research/RobotFarming/Projects/hyper_transformer/configs/config_HSIT_enmap_PRE_test.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    keys = read_csv_keys(
                        os.path.join(config["enmap_dataset"]["lmdb_save_dir"], 
                                     config["enmap_dataset"]["csv_file_name"]), 
                        config["enmap_dataset"]["columns"][0])
    # Create an instance of the dataset
    print(f"Number of patches: {len(keys)}")
    dataset = enmap_dataset(config, keys, is_train=True, is_dhp=False, want_DHP_MS_HR=False)

    # Get a sample from the dataset
    for i, sample in enumerate(dataset, 1):
        plt.imshow(sample[2], cmap='GnBu_r')
        plt.show()
        if i == 12:
            break
        