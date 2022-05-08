import numpy as np
import cv2
image1 = np.load('./results/04256520/cc644fad0b76a441d84c7dc40ac6d743_02_nerd.npz')
print(image1.f.w)

image2 = np.load('./results/04256520/cc644fad0b76a441d84c7dc40ac6d743_02_gt.npz')
print(image2.f.w)
# cv2.imshow("test", image1)