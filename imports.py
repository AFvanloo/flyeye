import flyeye as fly
import cv, cv2
import numpy as np
from matplotlib import pyplot as plt
import os

os.chdir('./eyes/bla3')

im = cv2.imread('./g1_1.tif')
hsv = cv2.cvtColor(im, cv.CV_BGR2HSV)

s = hsv[:,:,1]


os.chdir('../..')
