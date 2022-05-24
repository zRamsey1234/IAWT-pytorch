import torchvision.transforms as standard_transforms
import numpy as np
import os
import random
import glob
from torch.utils import data
from PIL import Image
import h5py
import cv2
import mmcv

class RGBIR(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, rgb_img_transform=None, gt_transform=None):
        self.data_root = data_path
        self.gt_root = data_path + '/dm_5'  # /home/zhoulin/cc/data/DroneRGBT/Test/dm_5
        print(self.gt_root)
        self.rgb_paths = []
        self.lwir_paths = []
        self.gt_paths = []
        for gt_path in glob.glob(os.path.join(self.gt_root, '*.h5')):
            self.rgb_paths.append(gt_path.replace('dm_5', 'Visible').replace('h5', 'jpg'))
            self.lwir_paths.append(gt_path.replace('dm_5', 'Infrared').replace('.h5', 'R.jpg'))
            self.gt_paths.append(gt_path)

        assert len(self.rgb_paths) == len(self.lwir_paths)
        assert len(self.gt_paths) == len(self.lwir_paths)
        self.num_samples = len(self.rgb_paths)

        self.main_transform = main_transform
        self.lwir_img_transform = img_transform
        self.gt_transform = gt_transform
        self.rgb_img_transform = rgb_img_transform

        self.mode = mode

    def __getitem__(self, index):
        rgb_img, lwir_img, den, fname = self.read_image_and_gt(index)
        if self.main_transform is not None:
            imgs, den = self.main_transform([rgb_img, lwir_img], den)
            rgb_img, lwir_img = imgs
        if self.lwir_img_transform is not None:
            lwir_img = self.lwir_img_transform(lwir_img)
        if self.rgb_img_transform is not None:
            rgb_img = self.rgb_img_transform(rgb_img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)

        if self.mode == 'train':
            rgb_img, lwir_img, den = self.random_crop(rgb_img, lwir_img, den)


        return rgb_img, lwir_img, den, fname

    def __len__(self):
        return self.num_samples


    def random_crop(self, img, infrared, den, dst_size=(512, 640)):
        # dst_size: ht, wd
        _, ts_hd, ts_wd = img.shape

        x1 = random.randint(0, ts_wd - dst_size[1])
        y1 = random.randint(0, ts_hd - dst_size[0])
        x2 = x1 + dst_size[1]
        y2 = y1 + dst_size[0]

        return img[:,y1:y2,x1:x2], infrared[:,y1:y2,x1:x2], den[y1:y2,x1:x2]

    def read_image_and_gt(self, index):
        # rgb_img = Image.open(self.rgb_paths[index])
        # lwir_img = Image.open(self.lwir_paths[index])
        rgb_img = Image.open(self.rgb_paths[index])
        lwir_img = Image.open(self.lwir_paths[index])
        if lwir_img.mode == 'L':
            lwir_img = lwir_img.convert('RGB')

        fname = self.rgb_paths[index].split('/')[-1].split('.')[0]
        gt_path = self.gt_paths[index]
        den = h5py.File(gt_path, 'r')
        den = np.asarray(den['density'])
        den = den.astype(np.float32)
        den = Image.fromarray(den)  # 这个是为啥

        return rgb_img, lwir_img, den, fname

    def get_num_samples(self):
        return self.num_samples

if __name__ == '__main__':
    rgb_mean_std = ([0.3489987, 0.33474705, 0.3524759], [0.2194258, 0.20472793, 0.21772078])
    mean_std = ([0.494935, 0.494935, 0.494935], [0.18174993, 0.18174993, 0.18174993])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    rgb_img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*rgb_mean_std)
    ])

    #
    # train_dataset = train_set = DroneRGBT('/home/zhoulin/cc/data/RGBT-CC-CVPR2021' + '/train', 'train',
    #                                   img_transform=img_transform, rgb_img_transform=rgb_img_transform,)
    # for i in range(10):
    #     rgb_img, lwir_img, den = train_dataset[i]
    #     print(rgb_img.shape)
    #     print(lwir_img.shape)

