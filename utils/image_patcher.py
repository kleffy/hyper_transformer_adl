import argparse
import os
from osgeo import gdal

class ImageSplitter:
    def __init__(self, src_path, dst_dir, patch_shape, strides, tile_prefix="Tile", black_pixel_threshold=(0, 23)):
        self.src_path = src_path
        self.dst_dir = dst_dir
        self.patch_shape = patch_shape
        self.strides = strides
        self.tile_prefix = tile_prefix
        self.black_pixel_threshold = black_pixel_threshold

    def split_image(self):
        gdal.UseExceptions()
        ds = gdal.Open(self.src_path)
        x_size = ds.RasterXSize
        y_size = ds.RasterYSize
        x_patch, y_patch = self.patch_shape
        x_stride, y_stride = self.strides
        skipped = 0
        for j in range(0, y_size, y_stride):
            if j + y_patch > y_size:
                j = y_size - y_patch
            for i in range(0, x_size, x_stride):
                if i + x_patch > x_size:
                    i = x_size - x_patch
                tile_name = "{}_{:05d}_{:05d}.Tif".format(self.tile_prefix, i, j)
                tile_path = os.path.join(self.dst_dir, tile_name)
                patch = ds.ReadAsArray(i, j, x_patch, y_patch)
                if patch.dtype == 'uint16':
                    black_pixels = (patch == 0).sum()
                    black_pixel_ratio = black_pixels / (x_patch * y_patch)
                    no_data = 0
                    if black_pixel_ratio > self.black_pixel_threshold[0]:
                        skipped += 1
                        # print(f"Skipping {tile_name} because it has too many black pixels.")
                        continue
                else:
                    black_pixels = (patch == -32768).sum()
                    black_pixel_ratio = black_pixels / (x_patch * y_patch)
                    no_data = -32768
                    if black_pixel_ratio > self.black_pixel_threshold[1]:
                        skipped += 1
                        # print(f"Skipping {tile_name} because it has too many black pixels.")
                        continue

                gdal.Translate(tile_path, ds, format='GTiff', srcWin=[i, j, x_patch, y_patch], noData=no_data,
                               options=['COMPRESS=DEFLATE'])
        print(f"Skipped {skipped} file(s).")

class DatasetSplitter:
    def __init__(self, src_dir, dst_dir, patch_shape, strides=None):
        self.src_dir = src_dir #'/'.join(src_dir.split('/')[:-1])
        self.dst_dir = dst_dir
        self.patch_shape = patch_shape
        self.strides = strides

    def split_dataset(self):
        src_names = sorted(os.listdir(self.src_dir))
        for src_name in src_names:
            src_path = os.path.join(self.src_dir, src_name)
            image_splitter = ImageSplitter(src_path, self.dst_dir, self.patch_shape, self.strides,
                                           src_name.split(".")[0])
            image_splitter.split_image()


if __name__ == '__main__':
    root_dir = r'/vol/research/RobotFarming/Projects/hyper_downloader/pre_process'
    out_dir = r'/vol/research/RobotFarming/Projects/hyper_transformer/datasets/enmap_gdal/n550_l12'
    parser = argparse.ArgumentParser(
        "Split data into tiles of desired height and width, with a specified stride.")
    parser.add_argument("--input_dir", required=False, default=root_dir,
                        help="Directory containing the imagery.")
    parser.add_argument("--output_dir", required=False, default=out_dir,
                        help="Directory where the tiles will be created.")
    parser.add_argument("--shape", type=int, default=160, help="Shape of the tile.")
    parser.add_argument("--stride", type=int, default=160, help="Length of the stride between tiles.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    tile_shape = (args.shape, args.shape)
    strides = (args.stride, args.stride)

    images = os.listdir(input_dir)
    
    ds = DatasetSplitter(input_dir, output_dir, tile_shape, strides)
    ds.split_dataset()
    print('Done!')