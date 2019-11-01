import cv2
import numbers
import random
from .cvEnhance import Color, Brightness, Contrast
import numpy as np


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

def _is_np_image(img):
    return type(img)==np.ndarray

def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (np.unit8): numpy ndarray with BGR channel order.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        numpy ndarray: Brightness adjusted image.
    """
    if not _is_np_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))
    enhancer = Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img

def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (np.unit8): numpy ndarray with BGR channel order.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        numpy ndarray: Contrast adjusted image.
    """
    if not _is_np_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))
    enhancer = Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img

def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (np.unit8): numpy ndarray with BGR channel order.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        numpy ndarray: Saturation adjusted image.
    """
    if not _is_np_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))
    enhancer = Color(img)
    img = enhancer.enhance(saturation_factor)
    return img

def adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        img (np.unit8): numpy ndarray with BGR channel order.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        numpy ndarray: Hue adjusted image.
    """
    if not _is_np_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))
    imghsv = img.copy().astype(np.uint8)
    imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2HSV)
    # note that the hue value range from 0 to 180, (hue in [0,180) )
    add_on = np.uint8(hue_factor * 180)
    with np.errstate(over='ignore'):
        imghsv[:,:,0] += add_on
    return cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        def _check_input(value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
            if isinstance(value, numbers.Number):
                if value < 0:
                    raise ValueError("If {} is a single number, it must be non negative.".format(name))
                value = [center - value, center + value]
                if clip_first_on_zero:
                    value[0] = max(value[0], 0)
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                if not bound[0] <= value[0] <= value[1] <= bound[1]:
                    raise ValueError("{} values should be between {}".format(name, bound))
            else:
                raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

            # if value is 0 or (1., 1.) for brightness/contrast/saturation
            # or (0., 0.) for hue, do nothing
            if value[0] == value[1] == center:
                value = None
            return value
        self.brightness = _check_input(brightness, 'brightness')
        self.contrast = _check_input(contrast, 'contrast')
        self.saturation = _check_input(saturation, 'saturation')
        self.hue = _check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        print(self.brightness)
    def __call__(self, img):
        img = img.astype(np.uint8)
        transforms = []
        brightness, contrast, saturation, hue = \
                self.brightness, self.contrast, self.saturation, self.hue
        
        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        for t in transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
