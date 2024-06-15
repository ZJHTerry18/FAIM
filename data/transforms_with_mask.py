import torch
import random
import math
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic, mask):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value), mask

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic), mask

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value), mask
        else:
            return img, mask


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, mask):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor, mask


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    def __init__(self, p = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.p = p
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img, mask):
        if random.uniform(0, 1) >= self.p:
            return img, mask

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img, mask

        return img, mask


class Resize(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size: Desired output size. If size is a sequence like
            (h, w), output size will be matched to this.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert len(size) == 2
        self.size = size
        self.h, self.w = size
        self.interpolation = interpolation

    def __call__(self, img, mask):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        resized_img = img.resize((self.w, self.h), self.interpolation)
        mh, mw = mask.size()
        if self.h == mh and self.w == mw:
            resized_mask = mask
        else:
            resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), self.size, mode='nearest').squeeze(0).squeeze(0)

        return resized_img, resized_mask


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.uniform(0, 1) >= self.p:
            return img, mask
        else:
            return img.transpose(Image.FLIP_LEFT_RIGHT), torch.flip(mask, [1])
        

class RandomCroping(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, p=0.5, interpolation=Image.BILINEAR):
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img, mask):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        width, height = img.size
        if random.uniform(0, 1) >= self.p:
            return img, mask
        
        new_width, new_height = int(round(width * 1.125)), int(round(height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (new_height, new_width), mode='nearest').squeeze()
        x_maxrange = new_width - width
        y_maxrange = new_height - height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + width, y1 + height))
        croped_mask = resized_mask[y1:y1+height, x1:x1+width]

        return croped_img, croped_mask


class RandomClothesErasing(object):
    """ 
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         mean: Erasing value. 
    """
    def __init__(self, p = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.p = p
        self.mean = mean
       
    def __call__(self, img, mask):
        if random.uniform(0, 1) < self.p:
            # erase upper clothes
            img[0][mask==2] = self.mean[0]
            img[1][mask==2] = self.mean[1]
            img[2][mask==2] = self.mean[2]
        if random.uniform(0, 1) < self.p:
            # erase pants
            img[0][mask==3] = self.mean[0]
            img[1][mask==3] = self.mean[1]
            img[2][mask==3] = self.mean[2]

        return img, mask

class RandomClothesErasingBoth(object):
    """ 
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         mean: Erasing value. 
    """
    def __init__(self, p = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.p = p
        self.mean = mean
       
    def __call__(self, img, mask):
        if random.uniform(0, 1) < self.p:
            # erase upper clothes
            img[0][mask==2] = self.mean[0]
            img[1][mask==2] = self.mean[1]
            img[2][mask==2] = self.mean[2]
            # erase pants
            img[0][mask==3] = self.mean[0]
            img[1][mask==3] = self.mean[1]
            img[2][mask==3] = self.mean[2]

        return img, mask

class RandomClothesErasingEither(object):
    """ 
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         mean: Erasing value. 
    """
    def __init__(self, p = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.p = p
        self.mean = mean
       
    def __call__(self, img, mask):
        if random.uniform(0, 1) < self.p:
            if random.uniform(0, 1) < 0.5:
                # erase upper clothes
                img[0][mask==2] = self.mean[0]
                img[1][mask==2] = self.mean[1]
                img[2][mask==2] = self.mean[2]
            else:
                # erase pants
                img[0][mask==3] = self.mean[0]
                img[1][mask==3] = self.mean[1]
                img[2][mask==3] = self.mean[2]

        return img, mask