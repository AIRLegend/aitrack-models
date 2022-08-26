import numpy as np
from PIL import Image
from PIL.Image import Image as ImageType
import torch
from torchvision import transforms as torchtransforms
from utils import get_roi

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
        
        if len(landmarks.shape) < 2:
            landmarks = landmarks.reshape((68, 2))
        
        if isinstance(image, torch.Tensor):
            image = (image).detach().numpy().squeeze()

        image = image.squeeze()

        w, h = image.shape[0], image.shape[1]

        ang = np.random.uniform(*self.rotation_range) 
        rotated_img = np.array(
            Image.fromarray(image).rotate(ang)
        )
        rotated_lms = self._to_homogeneous(landmarks)

        transform_mat = self._get_transform_mat(ang, origin=(w/2, h/2))
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
        
        image = torch.tensor(image) if isinstance(image, np.ndarray) else image

        if image.max() > 10:
            image = (image / 255.0)

        if len(image.shape) < 3:
            image = image.unsqueeze(0)

        
        image = self.norm(image)

        return {'image': image, 'landmarks': landmarks}



class RandomMirrorSample(object):
    def __init__(self, p=.5):
        self.p = p
        self.mirror_transform = torchtransforms.RandomHorizontalFlip(p=1)
        self.pil_to_tensor = torchtransforms.PILToTensor()
        
    def __call__(self, sample):
        if self.p < np.random.uniform():
            return sample

        image, landmarks = sample['image'], sample['landmarks']
        
        if isinstance(image, np.ndarray):
            image = torch.tensor(image)
        elif isinstance(image, ImageType):
            image = self.pil_to_tensor(image)

        image = self.mirror_transform(image)

        image_width = image.shape[-2]

        landmarks[:, 0] =  image_width - landmarks[:, 0]

        sample['image'] = image
        sample['landmarks'] = landmarks
        return sample


class CropROI(object):
    def __init__(self, range_zoomout=(.1, .3)):
        self.to_pil = torchtransforms.ToPILImage(mode='L')
        self.range_zoomout = range_zoomout
        
    def __call__(self, sample):
        """ image must be PIL image
        """
        image, landmarks = sample['image'], sample['landmarks']
        
        _, landmarks, crop = get_roi(
            image, landmarks, return_crop=True, 
            margin_percent=np.random.uniform(*self.range_zoomout)
        )

        sample['image'] = crop
        sample['landmarks'] = landmarks
        return sample


class RandomShift(object):
    def __init__(self, new_img_size=(114,114), p=.5):
        self.to_pil = torchtransforms.ToPILImage(mode='L')
        
        self.canvas_size= new_img_size
        self.p = p
        
    def __call__(self, sample):
        if np.random.uniform() > self.p:
            return sample
        
        image, landmarks = sample['image'], sample['landmarks']
        newlm = np.array(landmarks)
        
        if isinstance(image, torch.Tensor):
            im2 = self.to_pil(image)
        elif isinstance(image, np.ndarray):
            im2 = Image.fromarray(image.squeeze(), mode='L')
        else:
            im2 = image.copy()
            
            
        scale = np.random.uniform(0.5, 0.8)
        
        offset_x_percent = np.random.uniform(0.01, 0.3)
        offset_y_percent = np.random.uniform(0.01, 0.3)
            
        original_size = im2.size
        canvas_size = self.canvas_size
        
        paste_size = (int(canvas_size[0]*scale), int(canvas_size[1]*scale))
        
        offset = (int(canvas_size[0] * offset_x_percent), int(canvas_size[1] * offset_y_percent))
        
        black_canvas = Image.new("L", canvas_size)
        im2 = im2.resize(paste_size)
        black_canvas.paste(im2, offset)
        
        
        ratio_x, ratio_y = paste_size[0] / original_size[0], paste_size[1] / original_size[1]
        
        newlm[:, 0] *= ratio_x
        newlm[:, 1] *= ratio_y
        
        
        newlm[:, 0] += offset[0]
        newlm[:, 1] += offset[1]
        
        sample = {'image': np.array(black_canvas), 'landmarks': newlm}
        return sample


class Resize(object):
    def __init__(self, size=(114,114)):
        self.size = size
        
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        ratio_x, ratio_y = self.size[0] / image.size[0],  self.size[1] / image.size[1]
        
        image = image.resize(self.size)

        landmarks[:,0] = landmarks[:,0] * ratio_x
        landmarks[:,1] = landmarks[:,1] * ratio_y

        sample['image'] = image
        sample['landmarks'] = landmarks
        return sample



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
        
        if len(sample['image'].shape) < 3:
            sample['image'] = sample['image'].unsqueeze(0)

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