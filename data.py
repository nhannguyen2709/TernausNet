import argparse
import os
import numpy as np
import imageio
import pandas as pd

import torch

from torch.utils import data
from torchvision import transforms

from skimage.transform import resize


parser = argparse.ArgumentParser(description='PyTorch dataloader')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')


class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list, transform=None, test_mode=False):
        self.root_path = root_path
        self.file_list = file_list
        self.transform = transform
        self.test_mode = test_mode

    def _resize(self, image, num_channels):
        return resize(image, (128, 128, num_channels), mode='constant', preserve_range=True)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        file_id = self.file_list[index]
        
        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")
        
        if not self.test_mode:
            mask_folder = os.path.join(self.root_path, "masks")
            mask_path = os.path.join(mask_folder, file_id + ".png")
            image = self._resize(np.array(imageio.imread(image_path), dtype=np.uint8), 3)
            mask = self._resize(np.array(imageio.imread(mask_path), dtype=np.uint8), 1)
            processed_image, processed_mask = self.transform(image), self.transform(mask)
            return processed_image, processed_mask
        else:
            image = self._resize(np.array(imageio.imread(image_path), dtype=np.uint8), 3)
            processed_image = self.transform(image)
            return processed_image


# Debug
if __name__ == '__main__':
    args = parser.parse_args()

    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'test')
    train_file_list = pd.read_csv(os.path.join(args.data, 'train.csv'))['id']
    test_file_list = pd.read_csv(os.path.join(args.data, 'sample_submission.csv'))['id']
    
    train_dataset = TGSSaltDataset(traindir, train_file_list,
        transforms.ToTensor())
    test_dataset = TGSSaltDataset(testdir, test_file_list, 
        transforms.ToTensor(),
        test_mode=True)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)
    
    for i, (image, mask) in enumerate(train_loader):
        print(i, image.shape, mask.shape)

    # for i, image in enumerate(test_loader):
    #     print(i, image.shape)