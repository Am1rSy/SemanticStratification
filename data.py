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