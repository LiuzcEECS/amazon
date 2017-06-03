from __future__ import division
import math
import colorsys
from matplotlib import cm
import random
import pprint
from scipy import misc
from scipy.io import loadmat
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
from skimage.transform import rotate
import h5py
DATA_DIR = "./data/"
def data_augmentation(image, is_scale = True, is_rot = True, is_flip = True, is_crop = True,
is_color = True, is_contrast = True):
	res_image = np.array(image, copy = True)
	if is_scale:
		s = random.uniform(1,1.5)
		im = np.array([misc.imresize(image[i], (int(image.shape[1] * s), int(image.shape[2] * s))) for i in range(len(image))])
		res_image = np.concatenate((res_image, im[:,int((im.shape[1] - image.shape[1]) / 2):int((im.shape[1] - image.shape[1]) / 2) + image.shape[1],
		int((im.shape[2] - image.shape[2]) / 2):int((im.shape[2] - image.shape[2]) / 2) + image.shape[2],:]))
	if is_rot:
		s = random.uniform(-10.0, 10.0)
		res_image = np.concatenate((res_image, np.array([rotate(image[i], s, mode = 'edge' ,preserve_range = True) for i in range(len(image))])))
	if is_flip:
		res_image = np.concatenate((res_image, image[:,:,::-1,:]))
	if is_crop:
		pass
	if is_color:
		res_image = np.concatenate((res_image,image * random.uniform(0.8, 1.2)))
	if is_contrast:
		a = image.mean()
		s = random.uniform(0.8, 1.2)
		res_image = np.concatenate((res_image, np.clip(a + (image - a) * s, 0, 255)))
	return res_image

