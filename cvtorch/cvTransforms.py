import random
import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np
from .cvFunctional import ColorJitter as cvColorJitter

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        flag = target is None
        for t in self.transforms:
            image, target = t(image, target)
        if flag:
            return image
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (ow, oh)

    def __call__(self, image, target=None):
        size = self.get_size((image.shape[1], image.shape[0]))
        image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_LINEAR)
        if target is not None:
            target = target.resize((image.shape[1], image.shape[0]))
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = cv2.flip(image, 1).reshape(image.shape)
            if target is not None:
                target = target.transpose(0)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = cv2.flip(image, 0).reshape(image.shape)
            if target is not None:
                target = target.transpose(1)
        return image, target


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = cvColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target=None):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target=None):
        image = torch.from_numpy((image.astype(np.float32)).transpose((2, 0, 1)))
        return image, target

class NormalizeAsNumpy(object):
    ''' normalize a HxWxC image
    '''
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, target=None):
        image[...,:] = (image[...,:] - self.mean) / self.std
        return image.astype(np.uint8), target

class NormalizeAsTorch(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = torch.tensor(mean,dtype=torch.float32)
        self.std = torch.tensor(std,dtype=torch.float32)
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
                 
