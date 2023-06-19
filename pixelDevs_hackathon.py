import numpy as np
import cv2 as cv
from connected_components import get_connected_components
from matplotlib import pyplot as plt
from get_scale import get_scale

def get_segments(img):
  # Create boundaries for each segment
  segment1 = (img <= 70)                  # dark pollens
  segment2 = (img > 65) & (img <= 100)    # light pollens

  # Create a blank array where all the colors for each segment will be placed
  dark_segm = np.zeros((img.shape[0], img.shape[1], 3))
  light_segm = np.zeros((img.shape[0], img.shape[1], 3))

  # Assign colors to each segment
  dark_segm[segment1] = (0,0,1)
  light_segm[segment2] = (1,0,0)

  # Convert array segments to image
  plt.imsave("dark_segments.jpg", dark_segm)
  plt.imsave("light_segments.jpg", light_segm)

  # Load, save, and return the images
  dark_segm = cv.imread("dark_segments.jpg")
  light_segm = cv.imread("light_segments.jpg")
  return dark_segm, light_segm

def clean_segments(dark_segm, light_segm):

  dark_segm = cv.cvtColor(dark_segm , cv.COLOR_BGR2GRAY)
  light_segm = cv.cvtColor(light_segm , cv.COLOR_BGR2GRAY)

  # Clean the dark segments (morphological operations, gaussian blurring, threshold)
  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
  dark_segm_cleaned = cv.morphologyEx(dark_segm, cv.MORPH_OPEN, kernel, iterations=2)
  # dark_segm_cleaned = cv.morphologyEx(dark_segm_cleaned, cv.MORPH_CLOSE, kernel, iterations=2)
  dark_segm_cleaned = cv.erode(dark_segm_cleaned, kernel, iterations=1)
  dark_segm = cv.GaussianBlur(dark_segm_cleaned, (7, 7), 0)
  dark_segm_cleaned = cv.threshold(dark_segm_cleaned, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

  # Clean the light segments
  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
  light_segm_cleaned = cv.morphologyEx(light_segm, cv.MORPH_OPEN, kernel, iterations=3)
  light_segm_cleaned = cv.morphologyEx(light_segm_cleaned, cv.MORPH_CLOSE, kernel, iterations=3)
  light_segm = cv.GaussianBlur(light_segm_cleaned, (7, 7), 0)
  light_segm_cleaned = cv.threshold(light_segm_cleaned, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

  # return dark_segm,light_segm
  return dark_segm_cleaned, light_segm_cleaned

def count_pollens(segm_array):
  pass

if __name__ == "__main__":
  # Original images
  # img = cv.imread("practice_image_1.jpg", 0)
  img = cv.imread("practice_image_2.jpg", 0)

  # bgrtorgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

  # Get dark and light pollen segments (numpy array)
  dark_segm, light_segm = get_segments(img)

  # Clean the segment arrays
  dark_segm_cleaned, light_segm_cleaned = clean_segments(dark_segm, light_segm)

  # Display images
  titles = [
    "dark pollen segments",
    "light pollen segments",
    "dark pollen segments (cleaned)",
    "light pollen segments (cleaned)"
    ]
  images = [
    dark_segm,
    light_segm,
    dark_segm_cleaned,
    light_segm_cleaned
    ]
  for i in range(len(images)):
      plt.subplot(2,2, i+1), plt.imshow(images[i], "gray")
      plt.title(titles[i])
      plt.xticks([]), plt.yticks([])
  plt.show()
  
  dark_cc = get_connected_components(img, dark_segm_cleaned)[0]
  light_cc = get_connected_components(img, light_segm_cleaned)[0]
  um_to_pix = get_scale(img)

  titles = [
    "dark pollen segments (cleaned)",
    "light pollen segments (cleaned)",
    "dark connected components",
    "light connected components",
    ]
  images = [
    dark_segm_cleaned,
    light_segm_cleaned,
    dark_cc,
    light_cc,
    ]
  for i in range(len(images)):
      plt.subplot(2,2, i+1), plt.imshow(images[i], "gray")
      plt.title(titles[i])
      plt.xticks([]), plt.yticks([])
  plt.show()