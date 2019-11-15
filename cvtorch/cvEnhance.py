import cv2
import numpy as np

def blend(im1, im2, alpha):
    img = im1.astype(np.float64) * (1. - alpha) + im2.astype(np.float64) * alpha
    return np.clip(img, 0, 255).astype(np.uint8)

class _Enhance(object):
    def enhance(self, factor):
        """
        Returns an enhanced image.
        :param factor: A floating point value controlling the enhancement.
                       Factor 1.0 always returns a copy of the original image,
                       lower factors mean less color (brightness, contrast,
                       etc), and higher values more. There are no restrictions
                       on this value.
        :rtype: numpy array(image)
        """
        return blend(self.degenerate, self.image, factor)

class Contrast(_Enhance):
    def __init__(self, image):
        self.image = image.copy()
        mean = int(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).mean() + 0.5)
        imgd = (np.ones(image.shape[:2]) * mean).astype(np.uint8)
        self.degenerate = cv2.cvtColor(imgd, cv2.COLOR_GRAY2BGR)

class Brightness(_Enhance):
    def __init__(self, image):
        self.image = image.copy()
        self.degenerate = np.zeros(image.shape)

class Color(_Enhance):
    def __init__(self, image):
        self.image = image.copy()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        self.degenerate = np.stack([gray]*3, axis=2)
