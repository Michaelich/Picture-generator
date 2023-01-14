import numpy as np
import cv2

def mse(img1, img2):
    height, width = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    return err/(float(height*width))

def objective_function(img1, img2):
    im_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return mse(im_1, im_2)

def read_image(str):
    image1 = cv2.imread(str)
    image1 = cv2.cvtColor(np.float32(image1), cv2.COLOR_BGR2GRAY)
    return image1

#imageArray=read_image(source)