# read mat file and saved as images
import cv2
import numpy as np
import h5py
from matplotlib import pyplot

height = 460
width = 345

def extract_data(file_mat, image_dic):
	with h5py.File(file_mat,'r') as f:
		images = f['make3d_dataset_fchange/images'][:]

	image_num = len(images)
	for i in range(image_num):
		img = images[i,...].transpose((2, 1, 0))
		file = image_dic+str(i+1)+'.jpg'
		img = img*255
		img = img.astype('uint8')
		cv2.imwrite(file, img)
#		pyplot.imsave(file, img)

def extract_labels(file_mat, depth_dic):
	with h5py.File(file_mat,'r') as f:
		depths = f['make3d_dataset_fchange/depths'][:]

	depth_num = len(depths)
	for i in range(depth_num):
		img = depths[i,...].transpose((1, 0))
		file = depth_dic+str(i+1)+'.jpg'
		depth = img
		depth = depth.astype('uint8')
		cv2.imwrite(file, depth)
#		pyplot.imsave(file, img)

def main(argv=None):
	for focal_length in np.arange(460, 700, 40):
		file_mat = 'make3d_dataset_f'+str(focal_length)+'.mat'
		image_dic = 'make3d_dataset_f'+str(focal_length)+'/images/'
		depth_dic = 'make3d_dataset_f'+str(focal_length)+'/depths/'
		extract_data(file_mat, image_dic)
		extract_labels(file_mat, depth_dic)
		print 'f='+str(focal_length)+' dataset is converted!'
	print("Done!")

if __name__ == '__main__':
	main()
