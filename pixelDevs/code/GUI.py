import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QTextEdit, QPushButton, QDialog, QFileDialog, QLineEdit, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDir
from PyQt5 import uic

import numpy as np
import cv2 as cv
from connected_components import get_connected_components
from get_scale import get_scale
from matplotlib import pyplot as plt

class UI(QMainWindow):
  def __init__(self):
    super(UI, self).__init__()

    # attributes
    self.path = None
    self.dark_segm_cleaned = None
    self.light_segm_cleaned = None

    # load the UI file
    uic.loadUi("pixelDevs.ui", self)

    # define the widgets
    self.runButton = self.findChild(QPushButton, "browseImageButton")
    self.fileName = self.findChild(QLineEdit, "fileName")
    self.inputImage = self.findChild(QLabel, "inputImage")
    self.lightColoredImage = self.findChild(QLabel, "lightColoredImage")
    self.darkColoredImage = self.findChild(QLabel, "darkColoredImage")
    self.runButton = self.findChild(QPushButton, "runButton")
    self.lightPollensCountLabel = self.findChild(QLabel, "lightPollensCountLabel")
    self.darkPollensCountLabel = self.findChild(QLabel, "darkPollensCountLabel")
    self.lightPollensCount = self.findChild(QLabel, "lightPollensCount")
    self.darkPollensCount = self.findChild(QLabel, "darkPollensCount")
    self.conversion = self.findChild(QLabel, "conversion")

    # handling events in button
    self.browseImageButton.clicked.connect(self.browseFiles)
    self.runButton.clicked.connect(self.run)

    # show the app
    self.show()

  def browseFiles(self):
    # reset fields if browse button is clicked again
    self.lightPollensCount.setText('0')
    self.darkPollensCount.setText('0')
    self.inputImage.setPixmap(QPixmap(None))
    self.lightColoredImage.setPixmap(QPixmap(None))
    self.darkColoredImage.setPixmap(QPixmap(None))

    fname=QFileDialog.getOpenFileName(self, 'Open File', '', 'Image files (*.jpg *.jpeg *.png)')
    self.fileName.setText(fname[0])   # display file name
    self.inputImage.setPixmap(QPixmap(fname[0]))  # show image
    self.path = fname[0]

  
  def run(self):

    # Original images
    img = cv.imread(self.path, 0)
    # img = cv.imread("practice_image_2.jpg", 0)

    # Get dark and light pollen segments (numpy array)
    dark_segm, light_segm = self.get_segments(img)

    # Clean the segment arrays
    dark_segm_cleaned, light_segm_cleaned = self.clean_segments(dark_segm, light_segm)

    # get micrometer to pixel conversion
    um_to_pix = get_scale(img)

    # extract connected components to count pollem
    dark_cc, darkCount = get_connected_components(img, dark_segm_cleaned, um_to_pix, "Dark Pollen")
    plt.imsave("dark_cc.jpg", dark_cc)
    light_cc, lightCount = get_connected_components(img, light_segm_cleaned, um_to_pix, "Light Pollen")
    plt.imsave("light_cc.jpg", light_cc)


    # show generated images in GUI from path
    self.lightColoredImage.setPixmap(QPixmap('light_cc.jpg'))
    self.darkColoredImage.setPixmap(QPixmap('dark_cc.jpg'))

    # update the labels for pollens count
    self.lightPollensCount.setText(str(lightCount))
    self.darkPollensCount.setText(str(darkCount))

    # display pixel to micrometer conversion
    self.conversion.setText(str(um_to_pix))

  
  def get_segments(self, img):
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


  def clean_segments(self, dark_segm, light_segm):

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


if __name__ == '__main__':
  app = QApplication(sys.argv)
  UIWindow = UI()
  app.exec_()