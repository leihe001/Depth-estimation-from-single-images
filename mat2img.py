# read mat folder and saved as images

import cv2
import numpy as np
import h5py
from matplotlib import pyplot

height = 460
width = 345

def extract_data():
	with h5py.File('make3d_dataset_f460.mat','r') as f:
		images = f['make3d_dataset_fchange/images'][:]

	image_num = len(images)
	for i in range(image_num):
		img = images[i,...].transpose((2, 1, 0))
		file = 'make3d_dataset_f460/images/'+str(i+1)+'.jpg'
		img = img*255
		img = img.astype('uint8')
		cv2.imwrite(file, img)
#		pyplot.imsave(file, img)

def extract_labels():
	with h5py.File('make3d_dataset_f460.mat','r') as f:
		depths = f['make3d_dataset_fchange/depths'][:]

	depth_num = len(depths)
	for i in range(depth_num):
		img = depths[i,...].transpose((1, 0))
		file = 'make3d_dataset_f460/depths/'+str(i+1)+'.jpg'
		depth = img
		depth = depth.astype('uint8')
		cv2.imwrite(file, depth)
#		pyplot.imsave(file, img)

def main(argv=None):
	# Input  and groundtruth producer
	extract_data()
	extract_labels()
	print("Training data is converted into images!")

if __name__ == '__main__':
	main()