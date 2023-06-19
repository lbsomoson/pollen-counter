# REFERENCE: https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/

# import the necessary packages
import argparse
import cv2
from matplotlib import pyplot as plt

# get_connected_components(<original image>, <dark or light pollen segm image>)
def get_connected_components(img, pollen_segm, um_to_pix, pollen_type):

	output_img = img.copy()
	pollen_segm = cv2.bitwise_not(pollen_segm)

	# 4 or 8 connected
	connectivity = 4

	# threshold for minimum area, width, and height of ccs
	min_area = 250
	min_width = 50
	min_height = 50

	# initialization of largest cc variables
	largest_cc = 0
	largest_area = 0
	largest_box = [0,0,0,0]
	largest_output = img.copy()

	# apply connected component analysis
	output = cv2.connectedComponentsWithStats(
		pollen_segm, connectivity, cv2.CV_32S)
	(num_labels, labels, stats, centroids) = output
	new_num_labels = 0

	# loop over the number of unique connected component labels
	for i in range(0, num_labels):
		# get top-leftmost pixel location of cc
		x = stats[i, cv2.CC_STAT_LEFT]
		y = stats[i, cv2.CC_STAT_TOP]

		# get width, height, and area of cc
		w = stats[i, cv2.CC_STAT_WIDTH]
		h = stats[i, cv2.CC_STAT_HEIGHT]
		area = stats[i, cv2.CC_STAT_AREA]

		# in original image, draw green bounding boxes around significant ccs
		if w > min_width and h > min_height:

			# get largest cc
			if area > largest_area:
				largest_cc = i
				largest_area = area
				largest_w = w
				largest_h = h
				largest_box = [x,y,x+w,y+h]

			# get number of ccs above threshold
			new_num_labels+= 1

			# draw (green) bounding box
			cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
	
	print(f"Number of Connected Components", new_num_labels)

	# print largest pollen dimensions
	print(f"Largest {pollen_type}:")
	print(f"  Width (pix): {largest_w} ")
	print(f"  Height (pix): {largest_h} ")
	print(f"  Area (pix): {largest_area} ")
	print(f"  Width (micrometer): {largest_w/um_to_pix} ")
	print(f"  Height (micrometer): {largest_h/um_to_pix} ")
	print(f"  Area (micrometer): {largest_area/um_to_pix} ")

	return [output_img, new_num_labels]