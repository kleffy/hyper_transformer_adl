import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from enmap_dataset_gdal2 import PatchDataset as PD

class DataLoaderTester:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size

    def test_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        for batch in dataloader:
            patch = batch[0]
            patch = transforms.ToPILImage()(patch) # Convert patch to a PIL Image
            plt.imshow(patch)
            plt.show()


if __name__ == "__main__":
    # patch_extractor = PatchExtractor(img_dir='/path/to/images', patch_size=32, stride=16, output_csv='/path/to/patches.csv')
    # patch_extractor.extract_patches()
    csv_file =r"/vol/research/RobotFarming/Projects/hyper_transformer/csv_files/n1_patch_coordinates.csv"
    patch_dataset = PD(csv_file=csv_file, transform=transforms.ToTensor())

    dataloader_tester = DataLoaderTester(dataset=patch_dataset, batch_size=1)
    dataloader_tester.test_dataloader()
