import TensorFrost as tf
import numpy as np
import time
import os
import cv2
current_dir = os.path.dirname(os.path.abspath(__file__))

tf.initialize(tf.opengl)

img0 = "H:/ImgTest/0.jpg"
img1 = "H:/ImgTest/1.jpg"

# Load images into numpy arrays
img0 = cv2.imread(img0)
img1 = cv2.imread(img1)
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

size = img0.shape
radius = 4
size_down = (size[1] // radius, size[0] // radius)
# Gaussian blur the images with a kernel size of radius
img0 = cv2.GaussianBlur(img0, (3*radius+1, 3*radius+1), 0)
img1 = cv2.GaussianBlur(img1, (3*radius+1, 3*radius+1), 0)
# Resample images to 1/4 of their original size
img0 = cv2.resize(img0, size_down)
img1 = cv2.resize(img1, size_down)

#plot the images
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow(img0)
plt.subplot(1, 2, 2)
plt.imshow(img1)
plt.show()

