from skimage.metrics import structural_similarity
import cv2 as cv
import imutils
import os
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,help="first input image")
ap.add_argument("-s", "--second", required=True,help="second")
args = vars(ap.parse_args())

img1 = cv.imread(args["first"])
img2 = cv.imread(args["second"])

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned

(score, diff) = structural_similarity(gray1, gray2, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv.threshold(diff, 0, 255,cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    (x, y, w, h) = cv.boundingRect(c)
    cv.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
# show the output images
cv.imshow("Original", img1)
cv.imshow("Modified", img2)
#cv2.imshow("Diff", diff)
#cv2.imshow("Thresh", thresh)
cv.waitKey(0)