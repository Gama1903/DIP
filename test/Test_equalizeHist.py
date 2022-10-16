import pkg.EqualizeHist as EqualizeHist
import cv2 as cv

src = cv.cvtColor(
    cv.imread(cv.samples.findFile("Images\equalizeHist_orig.tif")),
    cv.COLOR_BGR2GRAY)
dst = EqualizeHist.equalizeHist_m(src)

cv.imshow("Original Image", src)
cv.imshow("Equalized Image", dst)
cv.waitKey()