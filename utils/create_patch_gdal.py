from osgeo import gdal
import os
import numpy as np
import pandas as pd

class PatchExtractor:
    
    def __init__(self, folder_path, patch_size=32, stride=16):
        self.folder_path = folder_path
        self.patch_size = patch_size
        self.stride = stride
    
    def extract(self):
        df = pd.DataFrame(columns=['filename', 'x', 'y'])
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.tif'):
                filepath = os.path.join(self.folder_path, filename)
                dataset = gdal.Open(filepath)
                x_size = dataset.RasterXSize
                y_size = dataset.RasterYSize
                for y in range(0, y_size-self.patch_size+1, self.stride):
                    for x in range(0, x_size-self.patch_size+1, self.stride):
                        patch = dataset.ReadAsArray(x, y, self.patch_size, self.patch_size)
                        if np.any(patch == 0):
                            continue
                        df = df.append({'filename': filename, 'x': x, 'y': y}, ignore_index=True)
        df.to_csv('patches.csv', index=False)


class PatchExtractor2:
    def __init__(self, folder_path, patch_size, overlap):
        self.folder_path = folder_path
        self.patch_size = patch_size
        self.overlap = overlap
        
    def extract_patches(self, output_csv_path):
        files = os.listdir(self.folder_path)
        patches = []
        for f in files:
            if f.endswith('.tif'):
                print(f'Extracting patches from {f}...')
                filepath = os.path.join(self.folder_path, f)
                dataset = gdal.Open(filepath)
                band = dataset.GetRasterBand(1)
                rows, cols = dataset.RasterYSize, dataset.RasterXSize
                for r in range(0, rows - self.patch_size + 1, self.patch_size - self.overlap):
                    for c in range(0, cols - self.patch_size + 1, self.patch_size - self.overlap):
                        patch = band.ReadAsArray(c, r, self.patch_size, self.patch_size)
                        if np.count_nonzero(patch == 0) == 0:
                            patches.append([f, r, c])
        
        df = pd.DataFrame(patches, columns=['filename', 'row', 'col'])
        df.to_csv(output_csv_path, index=False)
        print(f'CSV file saved to {output_csv_path}')


class PatchExtractor3:
    def __init__(self, img_dir, patch_size, stride, output_csv):
        self.img_dir = img_dir
        self.patch_size = patch_size
        self.stride = stride
        self.output_csv = output_csv

    def extract(self):
        img_files = [f for f in os.listdir(self.img_dir) if f.endswith('.TIF')]

        rows = []
        for img_file in img_files:
            # Open the image with gdal
            img_path = os.path.join(self.img_dir, img_file)
            img_ds = gdal.Open(img_path)

            # Get image size
            x_size = img_ds.RasterXSize
            y_size = img_ds.RasterYSize

            # Compute patch coordinates
            for j in range(0, y_size, self.stride):
                if j + self.patch_size > y_size:
                    j = y_size - self.patch_size

                for i in range(0, x_size, self.stride):
                    if i + self.patch_size > x_size:
                        i = x_size - self.patch_size

                    patch = img_ds.ReadAsArray(i, j, self.patch_size, self.patch_size)
                    # Check if patch contains any zero pixels
                    patch_has_black_pixels = (patch == 0).any()
                    if not patch_has_black_pixels:
                        rows.append((img_path, i, j, self.patch_size, self.stride))

        # Write output CSV
        df = pd.DataFrame(rows, columns=['filename', 'x', 'y', 'patch_size', 'stride'])
        df.to_csv(self.output_csv, index=False)

        print(f'Extracted {len(rows)} patches and saved to {self.output_csv}')

if __name__ == "__main__":
    img_dir = r"/vol/research/RobotFarming/Projects/hyper_downloader/test_data"
    patch_size = 160
    stride = 160
    output_dir = r"/vol/research/RobotFarming/Projects/hyper_transformer/csv_files"
    csv_filename = 'n1_patch_coordinates.csv'
    output_csv=os.path.join(output_dir, csv_filename)
    extractor = PatchExtractor3(img_dir=img_dir, patch_size=patch_size, stride=stride, output_csv=output_csv)
    extractor.extract()


# write a python class that loops through a folder of images and compute the coordinates of a patch and store the filename and the coordinates in a csv file. 
# the file is an hyperspectral image from EnMAP stored in a TIF format. use gdal to open the file and use pandas to open and write to the csv file. 
# The patches could be overlapping. I want to also skip a patch that has any pixel value equal to 0.