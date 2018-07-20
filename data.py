import argparse
import os
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

import torch

from torch.utils import data
from torchvision import transforms


parser = argparse.ArgumentParser(description='PyTorch dataloader')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')


def calc_padding(height, width):
    y_pad = 32 - height % 32
    y_min_pad = int(y_pad / 2)
    y_max_pad = y_pad - y_min_pad
    
    x_pad = 32 - width % 32
    x_min_pad = int(x_pad / 2)
    x_max_pad = x_pad - x_min_pad

    return x_min_pad, y_min_pad, x_max_pad, y_max_pad


class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list, transform=None, test_mode=False):
        self.root_path = root_path
        self.file_list = file_list
        self.transform = transform
        self.test_mode = test_mode

    def __len__(self):
        return len(self.file_list)
    
    def _load_image(self, image_path):
        return Image.open(image_path)
    
    def _load_mask(self, mask_path):
        return Image.open(mask_path).convert('L')

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        file_id = self.file_list[index]
        
        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")
        
        # mask_folder = os.path.join(self.root_path, "masks")
        # mask_path = os.path.join(mask_folder, file_id + ".png")
        
        # image = np.array(imageio.imread(image_path), dtype=np.uint8)
        # mask = np.array(imageio.imread(mask_path), dtype=np.uint8)
        if not self.test_mode:
            mask_folder = os.path.join(self.root_path, "masks")
            mask_path = os.path.join(mask_folder, file_id + ".png")
            image, mask = self._load_image(image_path), self._load_mask(mask_path)
            processed_image, processed_mask = self.transform(image), self.transform(mask)
            return processed_image, processed_mask
        else:
            image = self._load_image(image_path)
            processed_image = self.transform(image)
            return processed_image


# Debug
if __name__ == '__main__':
    args = parser.parse_args()

    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'test')
    train_file_list = pd.read_csv(os.path.join(args.data, 'train.csv'))['id']
    test_file_list = pd.read_csv(os.path.join(args.data, 'sample_submission.csv'))['id']
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    pad = transforms.Pad(padding=calc_padding(height=101, width=101))

    train_dataset = TGSSaltDataset(traindir, train_file_list,
        transforms.Compose([pad, transforms.ToTensor()]))
    test_dataset = TGSSaltDataset(testdir, test_file_list, 
        transforms.Compose([pad, transforms.ToTensor()]),
        test_mode=True)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)
    
    for i, (image, mask) in enumerate(train_loader):
        print(i, image.max(), mask.max())

    # for i, image in enumerate(test_loader):
    #     print(i, image.shape)