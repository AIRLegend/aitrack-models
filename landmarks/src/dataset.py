from torch.utils.data import Dataset, DataLoader
from utils import get_roi
import transforms as customtransforms
from torchvision import transforms as T
import torchfile
import torch
import numpy as np
from PIL import Image


class BWLS3D(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, 
            image_size=(114,114),
            crop_zoom_range=(.15, .3),
            transforms=None
        ):
        self.df = df
        self.image_size = image_size
        
        transf = [customtransforms.NormalizeSample()] if transforms is None else transforms
        transf = transf if not isinstance(transforms, list) else transforms
        self.transforms = T.Compose(transf)

        self.crop_transform = customtransforms.CropROI(crop_zoom_range)
        self.resize_transform = customtransforms.Resize(image_size)
        

    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.iloc[idx]['imgs']
        image = Image.open(img_path).convert('L')
        lms = torchfile.load(self.df.iloc[idx]['lms'])

        sample = {'image': image, 'landmarks': lms}


        #image, lms, _ = self._get_crop(image, lms)
        
        # image = np.array(
        #     image.resize(self.image_size)
        # )[None, :, :]

        # # Normalize image
        # image = image.astype(float) / 255.0
        
        sample = self.crop_transform(sample)
        sample = self.resize_transform(sample)

        sample['image'] =  np.array(sample['image'])
        if len(sample['image'].shape) < 3:
            sample['image'] = sample['image'][None, :, :] 


        sample = self.transforms(sample)

        if len(sample['landmarks'].shape) >= 2:
            sample['landmarks'] = sample['landmarks'].flatten()

        if sample['landmarks'].max() > 1:  # ensure targets are in 0-1 range
            sample['landmarks'] /= 114  # hardcoded, TODO: Change 


        lms = sample['landmarks']
        
        # include info of eye distance bb size for normalization
        # (after having transformations applied to the landmarks)
        eye1 = lms[39*2], lms[39*2] + 1
        eye2 = lms[42*2], lms[42*2] + 1
        sample['d'] = np.sqrt((eye2[0]-eye1[0])**2 + (eye2[1]-eye1[1])**2)

        return sample
