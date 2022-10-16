import pkg.MeanFilt as MeanFilt
import cv2 as cv

size_kernal = 11

src = cv.cvtColor(cv.imread(cv.samples.findFile("Images\\blurring_orig.tif")),
                  cv.COLOR_BGR2GRAY)
dst = MeanFilt.meanFilt_m(src, size_kernal)

cv.imshow("Original Image", src)
cv.imshow("Filtered Image", dst)
cv.waitKey()