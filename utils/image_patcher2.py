import os
import pandas as pd
from osgeo import gdal


class PatchGenerator:
    def __init__(self, input_dir, patch_size=64, stride=32):
        self.input_dir = input_dir
        self.patch_size = patch_size
        self.stride = stride
        self.df = pd.DataFrame(columns=['Image', 'Patch_Name', 'X1', 'Y1', 'X2', 'Y2'])
        self._generate_patch_coordinates()

    def _generate_patch_coordinates(self):
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.hdr'):
                dataset = gdal.Open(os.path.join(self.input_dir, filename))
                band = dataset.GetRasterBand(1)
                rows, cols = band.YSize, band.XSize

                for i in range(0, rows - self.patch_size + 1, self.stride):
                    for j in range(0, cols - self.patch_size + 1, self.stride):
                        x1, y1 = j, i
                        x2, y2 = j + self.patch_size, i + self.patch_size

                        patch_name = f"{filename.split('.')[0]}_{x1}_{y1}_{x2}_{y2}.npy"
                        self.df = self.df.append({'Image': filename, 'Patch_Name': patch_name,
                                                  'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2}, ignore_index=True)

    def save_patch_coordinates(self, csv_file):
        self.df.to_csv(csv_file, index=False)

    def read_patch(self, image_name, x1, y1):
        patch_df = self.df[(self.df['Image'] == image_name) &
                           (self.df['X1'] == x1) & (self.df['Y1'] == y1)]

        if patch_df.empty:
            raise ValueError(f"No patch found for image {image_name} and coordinates ({x1}, {y1})")

        patch_name = patch_df.iloc[0]['Patch_Name']
        px1, py1, px2, py2 = patch_df.iloc[0]['X1'], patch_df.iloc[0]['Y1'], \
                             patch_df.iloc[0]['X2'], patch_df.iloc[0]['Y2']

        patch_x1, patch_y1 = x1 % self.patch_size, y1 % self.patch_size
        patch_x2, patch_y2 = patch_x1 + self.patch_size, patch_y1 + self.patch_size

        dataset = gdal.Open(os.path.join(self.input_dir, image_name))
        band = dataset.GetRasterBand(1)
        patch = band.ReadAsArray(px1 + patch_x1, py1 + patch_y1, self.patch_size, self.patch_size)

        return patch, patch_name



# if __name__ == "__main__":
#     # Create a PatchExtractor object and extract patches from the input directory
#     extractor = PatchExtractor(input_dir='path/to/input/folder', patch_size=64, stride=32)
#     patch_coordinates = extractor.extract_patches()

#     # Save patch coordinates to CSV file
#     extractor.save_patch_coordinates_to_csv(output_path='patch_coordinates.csv')

#     # Load patch coordinates from CSV file
#     extractor = PatchExtractor.load_patch_coordinates_from_csv(input_path='patch_coordinates.csv')

#     # Get a patch from an image file
#     patch = extractor.get_patch(image_path='path/to/image/file', band_idx=1, x1=128, y1=256, x2=192, y2=320)

# # Get the patch coordinates from the patch_coordinates DataFrame
# patch_info = extractor.patch_coordinates.loc[(extractor.patch_coordinates['Image'] == 'image_file.hdr') & 
#                                               (extractor.patch_coordinates['Band'] == 1) & 
#                                               (extractor.patch_coordinates['X1'] == 128) & 
#                                               (extractor.patch_coordinates['Y1'] == 256) & 
#                                               (extractor.patch_coordinates['X2'] == 192) & 
#                                               (extractor.patch_coordinates['Y2'] == 320)]
# patch_name = patch_info['Patch_Name'].values[0]

# # Read the patch from the image file using the patch coordinates
# patch = extractor.get_patch(image_path='path/to/image/file', band_idx=1, x1=128, y1=256, x2=192, y2=320)

# Do something with the patch, such as pass it to a deep learning model for prediction

# import os
# import csv
# import numpy as np
# from osgeo import gdal

# # Specify input image directory and output patch directory
# input_dir = 'path/to/input/folder'
# output_dir = 'path/to/output/folder'

# # Set patch size and stride
# patch_size = 64
# stride = 32

# # Open the CSV file for writing
# with open('patch_coordinates.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Image', 'Patch_Name', 'X1', 'Y1', 'X2', 'Y2'])

#     # Loop through all image files in input directory
#     for filename in os.listdir(input_dir):
#         if filename.endswith('.hdr'):
#             # Open image file with GDAL
#             dataset = gdal.Open(os.path.join(input_dir, filename))
#             band = dataset.GetRasterBand(1)
#             rows, cols = band.YSize, band.XSize

#             # Compute the coordinates of all possible patches
#             for i in range(0, rows - patch_size + 1, stride):
#                 for j in range(0, cols - patch_size + 1, stride):
#                     x1, y1 = j, i
#                     x2, y2 = j + patch_size, i + patch_size

#                     # Create patch name based on image filename and patch coordinates
#                     patch_name = f"{filename.split('.')[0]}_{x1}_{y1}_{x2}_{y2}.npy"

#                     # Save patch coordinates to CSV file
#                     writer.writerow([filename, patch_name, x1, y1, x2, y2])

#                     # Extract the patch from the image and save to output directory
#                     patch = band.ReadAsArray(x1, y1, patch_size, patch_size)
#                     np.save(os.path.join(output_dir, patch_name), patch)
