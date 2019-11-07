from cvtorch.cvTransforms import Resize, RandomHorizontalFlip, RandomVerticalFlip, NormalizeAsNumpy, NormalizeAsTorch
from cvtorch.cvBox import BoxList
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

def resize_filp_test():
    img = cv2.imread('/core1/data/home/niuwenhao/data/tmp/02a0ae9a89b4a1937b228f0c1f90a8d8.jpg')
    box = np.array([[100,100,500,500],[600, 600, 400, 200]])
    box = BoxList(box, (img.shape[1], img.shape[0]), mode="xywh").convert("xyxy")
    
    for b in box.bbox:
        cv2.rectangle(img,(b[0], b[1]),(b[2], b[3]), (255,0,0), 3)
    cv2.imshow('pic', img)
    
    rsm = Resize(400, 600)
    imgr, boxr = rsm(img, box)
    
    imgrc = imgr.copy()
    
    for b in boxr.bbox:
        cv2.rectangle(imgr,(b[0], b[1]),(b[2], b[3]), (255,0,0), 3)
    cv2.imshow('picr', imgr)
    
    rhf = RandomHorizontalFlip(1)
    imghf,boxhf = rhf(imgrc, boxr)
    for b in boxhf.bbox:
        cv2.rectangle(imghf,(b[0], b[1]),(b[2], b[3]), (255,0,0), 3)
    cv2.imshow('pichf', imghf)
    
    rvf = RandomVerticalFlip(1)
    imgvf,boxvf = rvf(imgrc, boxr)
    for b in boxvf.bbox:
        cv2.rectangle(imgvf,(b[0], b[1]),(b[2], b[3]), (255,0,0), 3)
    cv2.imshow('picvf', imgvf)
    cv2.waitKey()

def normalize_test():
    box = np.array([[100,100,500,500],[600, 600, 400, 200]])
    norm = NormalizeAsNumpy(mean=[110, 107, 124], std=[1.,1.,1.])
    img = cv2.imread('/core1/data/home/niuwenhao/data/tmp/02a0ae9a89b4a1937b228f0c1f90a8d8.jpg')
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgn, boxn = norm(img, box)
    print(imgn.shape)
    imgpil = Image.open('/core1/data/home/niuwenhao/data/tmp/02a0ae9a89b4a1937b228f0c1f90a8d8.jpg')
    imgpil = F.to_tensor(imgpil).type(torch.float32)
    print(imgpil.shape)
   
    pnorm = NormalizeAsTorch(mean=[110, 107, 124], std=[1.,1.,1.])
    imgpiln, boxpn = pnorm(imgpil, box)
    
    imgn = imgn.astype(np.uint8)
    imgpiln = imgpiln.transpose(0,2).transpose(0,1).numpy().astype(np.uint8)
    
    cv2.imshow('cv', imgn)
    cv2.imshow('pil', imgpiln)
    cv2.waitKey()

if __name__=='__main__':
    resize_filp_test()
    normalize_test()
