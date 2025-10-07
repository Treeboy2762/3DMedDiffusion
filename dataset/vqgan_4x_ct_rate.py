import torch
from torch.utils.data.dataset import Dataset
import os
import random
import glob
import torchio as tio
import json
import pandas as pd
import numpy as np

class VQGANDataset_4x_CT_RATE(Dataset):
    def __init__(self, root_dir=None, augmentation=False, split='train', stage=1, patch_size=64):

        self.stage = stage
        self.split = split
        self.augmentation = augmentation
        
        with open(root_dir) as json_file:
            dataroots = json.load(json_file)
        
        # Handle CT-RATE specific structure with separate train/val paths
        if split == 'train':
            self.base_path = dataroots['CT_RATE_train']
            self.label = pd.read_csv(dataroots['CT_RATE_train_label'])
        elif split == 'val':
            self.base_path = dataroots['CT_RATE_val']
            self.label = pd.read_csv(dataroots['CT_RATE_val_label'])
        
        print(f"Found {len(self.label)} files for {split} split")
        print(f"Base path: {self.base_path}")
        
        self.patch_sampler = tio.data.UniformSampler(patch_size)
        # For chest CT: sample (256, 256, 128) to preserve both lungs in axial plane
        self.patch_sampler_256 = tio.data.UniformSampler((256, 256, 128))
        
        # Very minor affine translations for chest CT (conservative augmentation)
        self.minor_affine = tio.RandomAffine(
            scales=(0.98, 1.02),           # Very small scaling: ±2%
            degrees=(-3, 3),               # Very small rotation: ±2 degrees
            translation=(-4, 4),           # Small translation: ±3 voxels
            p=0.5                          # Apply 50% of the time
        )
        
        print(f'With patch size {str(patch_size)}')
        if augmentation:
            print('Minor affine augmentation enabled: scales=(0.98,1.02), rotation=±2°, translation=±3 voxels')
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        
        fname = self.label.loc[index, 'VolumeName']
        img_np = np.load(os.path.join('/tmp/gcsfuse_CTRATE/train', fname))[np.newaxis, :]

        # 2. Convert the NumPy array to a PyTorch tensor
        img_tensor = torch.from_numpy(img_np).float()

        # 3. Add a "channel" dimension. TorchIO expects 4D tensors: (C, D, H, W)
        #    Your .npy is likely 3D: (D, H, W), so we add a channel at the beginning.
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # 4. CORRECT WAY: Wrap the tensor in a tio.ScalarImage
        whole_img = tio.ScalarImage(tensor=img_tensor)

        # Apply minor affine augmentation if enabled (before final processing)
        if self.augmentation and self.split == 'train':
            whole_img = self.minor_affine(whole_img)

        if self.stage == 1 and self.split == 'train':
            img = None
            while img== None or img.data.sum() ==0:
                img = next(self.patch_sampler(tio.Subject(image = whole_img)))['image']
        elif self.stage ==2 and self.split == 'train':
            img = whole_img
        elif self.split =='val':
            img = whole_img
        

        
        imageout = img.data
        imageout = imageout.transpose(1, 3).transpose(2, 3)
        imageout = imageout.type(torch.float32)
        
        if self.split == 'val':
            return {'data': imageout, 'affine': img.affine, 'path': os.path.join('/tmp/gcsfuse_CTRATE/train', fname)} 
        else:
            return {'data': imageout}
