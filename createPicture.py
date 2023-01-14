import cv2
import numpy as np
from pandas.core.tools.datetimes import Scalar


def create_picture(chromosome, width, height):
    '''
    :param chromosome: made of circle
    :param circle: [x, y, radius, r, g, b]
    :param width: of picture
    :param height: of picture
    :return:
    '''
    picture = np.ones((width, height, 3), np.uint8)*255 # White background

    for cirlce in chromosome:
        cv2.circle(picture, (cirlce[0], cirlce[1]), cirlce[2], (cirlce[3], cirlce[4], cirlce[5]), -1)

    return picture

chromosome = [[0,0,10, 255, 0, 0],[50,50,10, 0, 255, 0], [99,105,10, 0, 0, 255]]

cv2.imshow('Picture',create_picture(chromosome, 100, 200))

cv2.waitKey(0)