import numpy as np
from PIL import Image
import torch
from torchvision import transforms as torchtransforms

class RandomRotation(object):
    def __init__(self, rotation_range=(-45.0, 45.0), rotation_prob=.3):
        self.rotation_range = rotation_range
        self.rotation_prob = rotation_prob
        
    def _to_homogeneous(self, vector):
        return np.hstack([
            vector, 
            np.ones(shape=(vector.shape[0], 1))
        ])
    
    def _get_transform_mat(self, angle, origin=(.5, .5)):
        transmat = np.array((
            (1, 0, -origin[0]),
            (0, 1, -origin[1]),
            (0, 0, 1)
        ))
        
        inv_transmat = np.linalg.inv(transmat)
        
        
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        rotmat = np.array(
            ((c, s, 0), 
             (-s, c, 0),
             (0, 0, 1),
            )
        )
        
        return inv_transmat @ rotmat @ transmat
    
        
    def __call__(self, sample):
        if np.random.uniform() > self.rotation_prob: 
            return sample
        
        image, landmarks = sample['image'], sample['landmarks']
        
        landmarks = landmarks.reshape((68, 2))

        h, w = image.shape[:2]
        
        if isinstance(image, torch.Tensor):
            image = (image).detach().numpy()

        image = image.squeeze()

        ang = np.random.uniform(*self.rotation_range) 
        rotated_img = np.array(
            Image.fromarray(np.uint8(image * 255)).rotate(ang)
        ) / 255
        rotated_lms = self._to_homogeneous(landmarks)

        transform_mat = self._get_transform_mat(ang)
        rotated_lms = (transform_mat @ rotated_lms.T).T

        return {
            'image': rotated_img[None, :, :], 
            'landmarks': rotated_lms[:, :2].flatten()
        }

    
class NormalizeSample(object):
    def __init__(self, 
                 mean=(0.485, 0.456, 0.406), 
                 std=(0.229, 0.224, 0.225)):
        
        if mean and std:
            self.norm = torchtransforms.Normalize(mean, std)
        else:
            self.norm = None
        
        
    def __call__(self, sample):
        if not self.norm:
            return sample

        image, landmarks = sample['image'], sample['landmarks']
        
        # image_size = image.shape[:2]

        if image.max() > 10:
            image = (image / 255)
            
        # if image.shape[0] != 3:
        #     image = np.transpose(image, (2,0,1))   # channels first
        
        image = torch.Tensor(image) if isinstance(image, np.ndarray) else image
        image = self.norm(image)

        return {'image': image, 'landmarks': landmarks}



class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class SampleImageTransform():
    def __init__(self, transform=None):
        self.transform = transform
    
    def __call__(self, sample):
        sample['image'] = torch.tensor(sample['image']) if isinstance(sample['image'], np.ndarray) else sample['image']
        sample['image'] = self.transform(sample['image'])

        return sample


class RandomContrastBrightness(SampleImageTransform):
    def __init__(self, transform=None, p=.1):
        super().__init__(None)
        self.transform = torchtransforms.ColorJitter(contrast=(.5, 1.6), brightness=(.5, 1.6)) if not transform else transform
        self.p = p

    def __call__(self, sample):
        if np.random.uniform() > self.p:
            return sample
        return super().__call__(sample)


class Posterize(SampleImageTransform):
    def __init__(self, transform=None, bits=4, p=.1):
        super().__init__(None)
        self.transform = torchtransforms.RandomPosterize(bits) if not transform else transform
        self.p = p

    def __call__(self, sample):
        if np.random.uniform() > self.p:
            return sample

        if isinstance(sample['image'], np.ndarray):
            sample['image'] = torch.tensor(sample['image'])

        if sample['image'].dtype != torch.uint8:
            if sample['image'].max() < 10:
                sample['image'] = (sample['image'] * 255).byte()

        sample =  super().__call__(sample)

        if sample['image'].max() > 10:  # revert back to [0-1] range
            sample['image'] = (sample['image'] / 255).float()

        return sample

class RandomSharpness(SampleImageTransform):
    def __init__(self, transform=None, p=.1):
        super().__init__(None)
        self.transform = torchtransforms.RandomAdjustSharpness(np.random.choice(12), p=p) if not transform else transform