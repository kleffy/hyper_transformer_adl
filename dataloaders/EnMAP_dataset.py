import collections
import math
import os
import random
from socket import MsgFlag

# import cv2
import numpy as np
import torch
from torch.utils import data

import scipy
import scipy.ndimage
import scipy.io

class enmap_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False, want_DHP_MS_HR=False
    ):
        self.split  = "train" if is_train else "val"        #Define train and validation splits
        self.config = config                                #Configuration file
        self.want_DHP_MS_HR = want_DHP_MS_HR                #This ask: DO we need DIP up-sampled output as dataloader output?
        self.is_dhp         = is_dhp                        #This checks: "Is this DIP training?"
        self.dir = self.config["enmap_dataset"]["data_dir"] #Path to enmap Center dataset 
        
        if self.split == "val":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
        elif self.split == "train":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
            if is_dhp:
                self.file_list = os.path.join(self.dir, f"{self.split}_dhp" + ".txt")
        
        self.images = [line.rstrip("\n") for line in open(self.file_list)] #Read image name corresponds to train/val/test set
    
        self.augmentation = self.config["enmap_dataset"]["augmentation"]   #Augmentation needed or not? 

        self.LR_crop_size = (self.config["enmap_dataset"]["LR_size"], self.config["enmap_dataset"]["LR_size"])  #Size of the the LR-HSI

        self.HR_crop_size = [self.config["enmap_dataset"]["HR_size"], self.config["enmap_dataset"]["HR_size"]]  #Size of the HR-HSI

        # cv2.setNumThreads(0)    # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])
    
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
        image_dict = self.files[self.split][index]
       
        # read each image in list
        mat         = scipy.io.loadmat(image_dict["imgs"])
        reference   = mat["ref"]
        PAN_image   = mat["pan"]

        if self.want_DHP_MS_HR:
            opt_lambda  = self.config["enmap_dataset"]["optimal_lambda"]
            mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            
            # Taking DIP up-sampled image as inputs
            MS_image = mat_dhp["dhp"]

            # Normalization
            #   MS_image    = torch.from_numpy(((np.array(MS_image) - self.dhp_mean)/self.dhp_std).transpose(2, 0, 1))
            #   PAN_image   = torch.from_numpy((np.array(PAN_image) - self.pan_mean)/self.pan_std)
            #   reference   = torch.from_numpy(((np.array(reference) - self.ref_mean)/self.ref_std).transpose(2, 0, 1))
        else:
            MS_image = mat["y"]
            
        # COnvert inputs into torch tensors
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))
        PAN_image   = torch.from_numpy(np.array(PAN_image)/1.0)
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))
        
        # Max Normalization
        MS_image    = MS_image/self.config["enmap_dataset"]["max_value"]
        PAN_image   = PAN_image/self.config["enmap_dataset"]["max_value"]
        reference   = reference/self.config["enmap_dataset"]["max_value"]           

        #If split = "train" and augment = "true" do augmentation
        if self.split == "train" and self.augmentation:
            MS_image, PAN_image, reference = self._augmentaion(MS_image, PAN_image, reference)

        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        return image_dict, MS_image, PAN_image, reference

    
    def __getitem__(self, index):

        image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)

        return image_dict, MS_image, PAN_image, reference