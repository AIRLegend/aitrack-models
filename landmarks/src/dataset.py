from torch.utils.data import Dataset, DataLoader
from utils import get_roi
from transforms import RandomRotation, NormalizeSample, UnNormalize
from torchvision import transforms as torchtransforms
import torchfile
import torch
import numpy as np
from PIL import Image


class BWLS3D(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, image_size=(114,114), transforms=None):
        self.df = df
        self.image_size = image_size
        
        transf = [NormalizeSample()] if transforms is None else transforms
        transf = transf if not isinstance(transforms, list) else transforms
        self.transforms = torchtransforms.Compose(transf)
        

    def __len__(self):
        return len(self.df)
    
    def _get_crop(self, image, lms):
        """Crops the face and landmarks so they're still aligned. """
        roi, points, crop = get_roi(image, lms, return_crop=True)
        
        points[:, 0] *= (self.image_size[0] / crop.size[0])
        points[:, 1] *= (self.image_size[1] / crop.size[1])
        
        return crop, points, roi
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.iloc[idx]['imgs']
        image = Image.open(img_path).convert('L')
        lms = torchfile.load(self.df.iloc[idx]['lms'])

        image, lms, _ = self._get_crop(image, lms)
        
        image = np.array(
            image.resize(self.image_size)
        )[None, :, :]

        image = image.astype(float) / 255.0
        lms = (lms/114).flatten()

        sample = {'image': image, 'landmarks': lms}
    
        sample = self.transforms(sample)

        # include info of eye distance bb size for normalization
        # (after having transformations applied to the landmarks)
        eye1 = lms[39*2], lms[39*2] + 1
        eye2 = lms[42*2], lms[42*2] + 1
        sample['d'] = np.sqrt((eye2[0]-eye1[0])**2 + (eye2[1]-eye1[1])**2)

        return sample
