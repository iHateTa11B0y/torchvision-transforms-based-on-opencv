from cvtorch.cvFunctional import ColorJitter as cvColorJitter
from torchvision.transforms import ColorJitter
import cv2
from PIL import Image
import matplotlib.pyplot as plt

imgf = '/core1/data/home/niuwenhao/data/tmp/f9a33c6de7f1c246114558bd411a0425.jpg'

brightness=0.3
contrast=0.4
saturation=0.2
hue=0.3

cvimg = cv2.imread(imgf)
cvJ = cvColorJitter(brightness, contrast, saturation, hue)
img = Image.open(imgf)
J = ColorJitter(brightness, contrast, saturation, hue)

plt.figure("origin")
plt.imshow(img)

img_jitter = J(img)

plt.figure("jitter")
plt.imshow(img_jitter)
plt.show()

cv2.imshow('origin_cv', cvimg)
cvimg_jitter = cvJ(cvimg)
cv2.imshow('jitter_cv', cvimg_jitter)
cv2.waitKey()



