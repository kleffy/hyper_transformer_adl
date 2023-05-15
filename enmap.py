# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 21:04:51 2022

@author: Administrator
"""

import os
from matplotlib import patches
import rasterio as rio
import csv
from rasterio.enums import Resampling
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from patchify import patchify
from PIL import Image
from skimage import io
from random import seed, sample, shuffle
from scipy.io import savemat


l2a_path = r'/vol/research/RobotFarming/Projects/hyper_downloader/tif_images_2A/ENMAP01-____L2A-DT0000002071_20220801T074432Z_002_V010111_20230124T141313Z-SPECTRAL_IMAGE.TIF'
l1_path = r'/vol/research/RobotFarming/Projects/data/clipped/ENMAP01-____L1C-DT0000001724_20220718T194048Z_022_V010111_20230124T201431Z-SPECTRAL_IMAGE.tif'


class EnmapBook:
    def __init__(self) -> None:
        pass

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)

        if not os.path.exists(directory):
            os.makedirs(directory)


    def read_file(self, file_path) -> np.ndarray:
        # img = rio.open(file_path) 
        # data = img.read()
        data = io.imread(file_path) # -> CxHxW
        return data #np.moveaxis(data, 0, -1) 


    def save_data(self, data, name, path=None):
        # data = self.extract_percentile_range(data, 2, 98)
        data = self.move_axis(data, channel_last=False)
        pan_image = self.extract_pan(data=data, pan_range=(1, 51))

        rgb_image = self.extract_rgb(data=data, r_range=(46, 48), g_range=(23, 25), b_range=(8, 10))
        
        # dt = (data - np.percentile(data, 2))/np.percentile(data, 98)
        data_dict = {name : self.move_axis(data, channel_last=True),
                    f'{name}_rgb': self.move_axis(rgb_image, channel_last=True),
                    'pan': self.move_axis(pan_image, channel_last=True),
                    # 'l1b_vnir_raw': l1b_vnir_img_data,
                    # 'lib_swir_raw': l1b_swir_img_data,
                    # 'l2a_raw': l2a_img_data
                    }
        if path:
            name = os.path.join(path, name)
        savemat(f'{name}_dataset.mat', data_dict)


    def extract_rgb(self, data, r_range:tuple, g_range:tuple, b_range:tuple) -> np.ndarray:
        # print(f'extract_rgb - data shape:: {data.shape}')
        # data = data.cpu().numpy().squeeze()
        r_mean = np.mean(data[r_range[0] : r_range[-1], :, :], axis=0)
        g_mean = np.mean(data[g_range[0] : g_range[-1], :, :], axis=0)
        b_mean = np.mean(data[b_range[0] : b_range[-1], :, :], axis=0)

        rgb_img = np.zeros((3, data.shape[1], data.shape[2]))

        rgb_img[0, :, :] = r_mean
        rgb_img[1, :, :] = g_mean
        rgb_img[2, :, :] = b_mean
        
        # rgb_img = (rgb_img - np.min(rgb_img))/np.ptp(rgb_img)
        # print(f'After: {np.max(rgb_img)}')
        return rgb_img


    def extract_pan(self, data:np.ndarray, pan_range:tuple) -> np.ndarray:
        dt = data[pan_range[0]:pan_range[-1],:,:]
        print(f'Before: {np.max(dt)}')

        # dt = (dt - np.percentile(dt, 2))/np.percentile(dt, 98)
        
        # dt = self.stretch_minmax(dt, 0, np.max(dt))
        pan_mean = np.mean(dt, axis=0)
        print(f'Pan Shape: {pan_mean.shape}')
        
        print(f'After: {np.max(pan_mean)}')
        return pan_mean


    def scale_image(self, path, scale_factor=1) -> np.ndarray:
        with rio.open(path) as dataset:
            # resample data to target shape
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * scale_factor),
                    int(dataset.width * scale_factor)
                ),
                resampling=Resampling.bilinear
            )
            print(dataset.count)
            show(dataset)

        return np.moveaxis(data.read(), 0, -1)


    def create_pan(self, data, pan_range:tuple) -> np.ndarray:
        p_mean = np.mean(data[:,:,pan_range[0]:pan_range[-1]], axis=2)
        # pan = (p_mean - np.min(p_mean))/np.ptp(p_mean)
        return p_mean


    def create_grayscale(self, rgb_image:np.ndarray) -> np.ndarray:
        gray = 0.216 * rgb_image[:,:,0] + 0.715 * rgb_image[:,:,1] + 0.0722 * rgb_image[:,:,2]
        return gray


    def generate_train_test_split(self, data, fraction:float=0.2):
        assert fraction >= 0 and fraction <= 1, 'The fraction must be between 0 and 1. Note, 0.2 implies that train set is 80% while test set is 20%' 
        seed(1)
        shuffle(data)
        stop_index = round(len(data) * fraction)
        train, test = data[stop_index:], data[:stop_index]
        print(f'Train - {len(train)}, Test - {len(test)}')
        return train, test

    
    def save_file_names(self, data, path=None):
        for name, entries in data.items():
            if path:
                name = os.path.join(path, name)

            with open(name + '.txt', 'w') as fp:
                for item in entries:
                    # write each item on a new line
                    fp.write("%s\n" % item)
        print('Done')


    # def create_patches(self, data:np.ndarray, patch_size:tuple, step_size:int, path:str=None) -> None:
    #     file_names = []
    #     assert 1 == 0, 'did you uncomment line 123?'
    #     for image in range(data.shape[-1]):
    #         large_image = data[:,:,image]
    #         # print(f'Large Image shape => {large_image.shape}')
    #         img_patches = patchify(large_image, patch_size, step=step_size)
    #         # print(f'img_patches shape => {img_patches.shape}')

    #         for i in range(img_patches.shape[0]):
    #             for j in range(img_patches.shape[1]):
    #                 patch = img_patches[i, j,:,:]
    #                 # print(f'Patch Image shape => {patch.shape}')
    #                 file_name = f"image_{str(i)}_{str(j)}.tiff"
    #                 file_names.append(file_name)
    #                 if path:
    #                     file_name = os.path.join(path, file_name)
    #                 # io.imsave(file_name, patch)
        
    #     return file_names
        


    def display_image(self, image, save_image=False, path=None, fname='rgb_color', gray=False) -> None:
        plt.figure(figsize=(22, 18))
        plt.axis('off')
        plt.imshow(image, cmap='gray') if gray else plt.imshow(image)
        plt.show()
        if save_image:
            if path:
                fname = os.path.join(path, fname)
            plt.savefig(f'{fname}.png')


    def get_img_within_percentile(self, img, lower, upper):
        img_ = (img - np.percentile(img, lower))/np.percentile(img, upper)
        return img_
    
    
    def extract_percentile_range(self, data, lo, hi):
        plo = np.percentile(data, lo)
        phi = np.percentile(data, hi)
        data[data[:,:,:] < plo] = plo
        data[data[:,:,:] >= phi] = phi
        data = (data - plo) / (phi - plo) 
        return data


    def stretch_minmax(self, val, min, max):
        return (val - min)/(max - min)
        

    def move_axis(self, data, channel_last:bool=False):
            if channel_last:
                data = np.moveaxis(data, 0, -1)
            else:
                data = np.moveaxis(data, (1, 0), (2, 1))
            
            return data

enmap = EnmapBook()

# path=r'adl/patches/images/'

# train_test_fpath=r'adl/patches/'

# belen_path = r'datasets/belen/L2A_Arcachon_Bands1To51_MEAN'

savemat_path = r'/vol/research/RobotFarming/Projects/hyper_transformer/datasets/enmap_data'

img_raw = enmap.read_file(l1_path)
# maxx = np.max(img_raw, axis=(1,2))
print(img_raw.shape)
# img_data = enmap.extract_percentile_range(img_raw, 10, 70)

enmap.save_data(img_raw, 'enmap', savemat_path)

# rgb_image = enmap.extract_rgb(data=img_raw, r_range=(46, 48), g_range=(23, 25), b_range=(8, 10))

# pan = enmap.extract_pan(data=img_raw, pan_range=(1, 51))

# pan = enmap.extract_pan_nomean(data=img_raw)

# pan = enmap.create_pan(data=rgb_image, pan_range=(1,2))

# pan = enmap.create_grayscale(rgb_image)

# rgb_image = enmap.stretch_minmax(rgb_image, 0, 255)

# pan = enmap.stretch_minmax(pan, 0, np.max(pan))
# rgb_image = (rgb_image - np.percentile(rgb_image, 2))/np.percentile(rgb_image, 98)
# rgb_image = enmap.move_axis(rgb_image, channel_last=True)
# print(rgb_image.shape)
# enmap.display_image(rgb_image)

# enmap.display_image(pan, gray=True)

# enmap.display_image(pan)

# file_names = enmap.create_patches(img_raw, patch_size=(160, 160), step_size=16, path=path)
# path = r'/vol/research/RobotFarming/Projects/hyper_transformer/datasets/enmap_data/enmap'
# file_names = os.listdir(path)
# train, test = enmap.generate_train_test_split(file_names, 0.2)

# name_dict = {'train': train, 'test': test}

# enmap.save_file_names(name_dict, path)
print('Done!')
