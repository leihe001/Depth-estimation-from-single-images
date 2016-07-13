import cv2
import numpy as np
from matplotlib import pyplot

sour_image_path = './'
list_file  = 'list.txt'
height = 172
width = 230

dest_image_path = '/home/lei/Downloads/Test134/'

dest_image_name_list = [] # read destination image
with open(dest_image_path + list_file) as fid:
	dest_image_name_list = [x.strip() for x in fid.readlines()]
dest_image_num = len(dest_image_name_list)

sour_image_name_list = [] # read source image
with open(sour_image_path + list_file) as fid:
	sour_image_name_list = [x.strip() for x in fid.readlines()]
source_image_num = len(sour_image_name_list)

for i in range(source_image_num):
	sour_img = cv2.imread(sour_image_name_list[i])
	sour_img = cv2.resize(sour_img, (height, width))
	hog = cv2.HOGDescriptor()
	sour_h = hog.compute(sour_img)
	i_dict = {} # set up empty dictionary
	for idx in range(dest_image_num):
		dest_img = cv2.imread(dest_image_path + dest_image_name_list[idx])
		dest_img = cv2.resize(dest_img, (height, width))
	 	dest_h = hog.compute(dest_img)
	 	dis_value = np.sqrt(np.sum((dest_h - sour_h)**2)) # compare the distance of the two images
	 	i_dict[dis_value] = idx	
	sorted_i_dict = sorted(i_dict) # sorted the distance
	print i_dict[sorted_i_dict[0]], i_dict[sorted_i_dict[1]], i_dict[sorted_i_dict[2]]
	i_dict.clear()  # clear the value of dict


