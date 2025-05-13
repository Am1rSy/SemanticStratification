"""
Dataset classes used to train segmentation models.
"""
import json
import os
import re
from glob import glob
from typing import Literal, Optional, Callable, Union
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
Image.MAX_IMAGE_PIXELS = None

class CityScapesDataset(Dataset):
    def __init__(self, path: str, split: Literal['train', 'val', 'test'],
                 image_transform: Optional[Callable] = None, annotation_transform: Optional[Callable] = None):
        self.num_channels = 3
        self.num_classes = 34
        # Store transforms for later
        self.image_transform = image_transform
        self.annotation_transform = annotation_transform
        # Load images
        self.img_dir = os.path.join(path, 'leftImg8bit_trainvaltest', 'leftImg8bit', split)
        assert os.path.exists(self.img_dir)
        self.img_files = [os.path.join(loc, f) for loc in os.listdir(self.img_dir) for f in
                          os.listdir(os.path.join(self.img_dir, loc))]
        self.img_files = [os.path.join(self.img_dir, f) for f in self.img_files]
        self.img_files = sorted(self.img_files)
        # Load annotations
        self.ann_dir = os.path.join(path, 'gtFine_trainvaltest', 'gtFine', split)
        assert os.path.exists(self.ann_dir)
        self.ann_files = [os.path.join(loc, f) for loc in os.listdir(self.ann_dir) for f in
                          os.listdir(os.path.join(self.ann_dir, loc)) if f.endswith('labelIds.png')]
        self.ann_files = [os.path.join(self.ann_dir, f) for f in self.ann_files]
        self.ann_files = sorted(self.ann_files)
        assert len(self.img_files) == len(self.ann_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        image = Image.open(self.img_files[idx])
        if self.image_transform:
            image = self.image_transform(image)
        annotation = Image.open(self.ann_files[idx])
        if self.annotation_transform:
            annotation = self.annotation_transform(annotation)
        return image, annotation
    

class CamVidDataset(Dataset):
    def __init__(self, path: str, split: Literal['train'],
                 image_transform: Optional[Callable] = None, annotation_transform: Optional[Callable] = None):
        self._num_channels = 3
        self._num_classes = 32
        # Store transforms for later
        self.image_transform = image_transform
        self.annotation_transform = annotation_transform
        
        ## this assumes prerpoc script was already used
        # Load images imags and annotations
        self.img_files, self.ann_files = self._get_png_and_jpg_files(os.path.join(path, "camvid"))
        
        assert len(self.img_files) == len(self.ann_files)
    
    @property
    def num_channels(self):
        return self._num_channels
    
    @property
    def num_classes(self):
        return self._num_classes


    def _get_png_and_jpg_files(self,
                              directory: str) -> tuple[list[str]]:
        png_files = []
        jpg_files = []

        for filename in os.listdir(directory):
            if filename.lower().endswith(".png"):
                png_files.append(os.path.join(directory, filename))
            elif filename.lower().endswith((".jpg", ".jpeg")):
                jpg_files.append(os.path.join(directory, filename))

        return jpg_files, png_files
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        image = Image.open(self.img_files[idx])
        if self.image_transform:
            image = self.image_transform(image)
        annotation = Image.open(self.ann_files[idx]).convert("L")
        if self.annotation_transform:
            annotation = self.annotation_transform(annotation)
        
        resize_transform = transforms.Resize((1024, 1024))

        image, mask = resize_transform(image), resize_transform(annotation)

        return image, mask

class PascalVOCDataset(Dataset):
    def __init__(self, path: str, split: Literal['train'],
                 image_transform: Optional[Callable] = None, annotation_transform: Optional[Callable] = None):
        self.num_channels = 3
        self.num_classes = 21
        # Store transforms for later
        self.image_transform = image_transform
        self.annotation_transform = annotation_transform
        # Load images
        self.img_dir = os.path.join(path, 'pascalvoc', 'images')
        assert os.path.exists(self.img_dir)
        self.img_files = [os.path.join(self.img_dir, loc) for loc in os.listdir(self.img_dir)]
        # self.img_files = [os.path.join(self.img_dir, f) for f in self.img_files]
        self.img_files = sorted(self.img_files)
        # Load annotations
        self.ann_dir = os.path.join(path, 'pascalvoc', 'masks')
        assert os.path.exists(self.ann_dir)
        self.ann_files = [os.path.join(self.ann_dir, loc) for loc in os.listdir(self.ann_dir)]
        self.ann_files = sorted(self.ann_files)
        assert len(self.img_files) == len(self.ann_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        image = Image.open(self.img_files[idx])
        if self.image_transform:
            image = self.image_transform(image)
        annotation = Image.open(self.ann_files[idx])
        if self.annotation_transform:
            annotation = self.annotation_transform(annotation)
        resize_transform = transforms.Resize((256, 512))

        image, mask = resize_transform(image), resize_transform(annotation)

        return image, mask

class EndoVisDataset(Dataset):
    def __init__(self, path: str, split: Literal['train', 'val', 'test'],
                 image_transform: Optional[Callable] = None, annotation_transform: Optional[Callable] = None):
        self.num_channels = 3
        self.num_classes = 12

        # Store transforms for later
        self.image_transform = image_transform
        self.annotation_transform = annotation_transform

        # Load categories
        with open(os.path.join(path, split, 'labels.json'), 'r') as f:
            self.categories = json.load(f)

        # Load images
        self.img_files = sorted(glob(os.path.join(path, split, 'seq_*/left_frames/*.png')))

        # cache data to RAM
        self.cache = []
        from tqdm import tqdm
        for path_img in tqdm(self.img_files):
            path_ann = path_img.replace('left_frames', 'labels')
            # image = Image.open(path_img)
            mask_color = np.array(Image.open(path_ann).convert('RGB'))
            mask_gray = np.zeros((1024, 1280), np.int8)

            for cat in self.categories:
                if cat['classid'] == 0:
                    continue
                mask = np.all(mask_color == np.array(cat['color']), axis=-1)
                mask_gray[mask] = cat['classid']

            # self.cache.append((image, Image.fromarray(mask_gray, mode='L')))
            self.cache.append(Image.fromarray(mask_gray, mode='L'))

        print('Dataset EndoVis2018Dataset is cached.')

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # image, annotation = self.cache[idx]
        image = Image.open(self.img_files[idx])
        annotation = self.cache[idx]

        if self.image_transform:
            image = self.image_transform(image)

        if self.annotation_transform:
            annotation = self.annotation_transform(annotation)
        
        return image, annotation

class LoveDADataset(Dataset):
    def __init__(self, path: str, split: Literal['train', 'val', 'test'],
                 image_transform: Optional[Callable] = None, annotation_transform: Optional[Callable] = None):
        self.num_channels = 3
        self.num_classes = 8
        # Store transforms for later
        self.image_transform = image_transform
        self.annotation_transform = annotation_transform
        # Base directory with images and annotations
        self.base_dir = os.path.join(path, split.title(), split.title())
        assert os.path.exists(self.base_dir)
        # Load images
        self.img_files = [os.path.join(loc, 'images_png', f) for loc in ['Rural', 'Urban'] for f in
                          os.listdir(os.path.join(self.base_dir, loc, 'images_png'))]
        self.img_files = [os.path.join(self.base_dir, f) for f in self.img_files]
        self.img_files = sorted(self.img_files)
        # Load annotations
        self.ann_files = [os.path.join(loc, 'masks_png', f) for loc in ['Rural', 'Urban'] for f in
                          os.listdir(os.path.join(self.base_dir, loc, 'masks_png'))]
        self.ann_files = [os.path.join(self.base_dir, f) for f in self.ann_files]
        self.ann_files = sorted(self.ann_files)
        assert len(self.img_files) == len(self.ann_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        image = Image.open(self.img_files[idx])
        if self.image_transform:
            image = self.image_transform(image)
        annotation = Image.open(self.ann_files[idx])
        if self.annotation_transform:
            annotation = self.annotation_transform(annotation)
        return image, annotation
