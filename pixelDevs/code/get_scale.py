import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def get_scale(img):
  # Crop the image to get the box that has the conversion scale
  # cropped = img[1950:2070,3370:3735]  # hardcoded

  # Apply threshold to remove cells and retain white box
  threshold1 = cv.threshold(img, 250, 255, cv.THRESH_BINARY)[1]

  # Apply otsu threshold to extract measurement more accurately
  threshold2 = cv.threshold(threshold1, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

  # Extract connected components
  analysis = cv.connectedComponentsWithStats(threshold2,4,cv.CV_32S)
  (totalLabels, label_ids, values, centroid) = analysis

  # Get largest width (excluding background and white rectangle) for pixel to micrometer conversion
  all_widths = values[:,cv.CC_STAT_WIDTH]
  max_width = np.sort(all_widths)[-3]
  pix_to_um = max_width/100
  
  # print(np.sort(all_widths))
  # print(pix_to_um)

  return pix_to_um
  
  # Loop through each component to create a mask for the scale lines
  # for i in range(1, totalLabels):
  #   area = values[i, cv.CC_STAT_AREA]
  #   x = values[i, cv.CC_STAT_LEFT]
  #   y = values[i, cv.CC_STAT_TOP]
  #   w = values[i, cv.CC_STAT_WIDTH]
  #   h = values[i, cv.CC_STAT_HEIGHT]
  #   print(w)
  #   cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
  #   # componentMask = (label_ids == i).astype("uint8") * 255
  #   # output = cv.bitwise_or(output, componentMask)

  # plt.imshow(img)
  # plt.show()