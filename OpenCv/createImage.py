import numpy as np  
import cv2 as cv  
im = cv.imread(r'Data/cat/image4.jpg')  
# print(im[0][0][0])
im[0][0][0] = 23
# print(im[0][0][0])
# cv.imshow("im",im)
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)  

ret, thresh = cv.threshold(imgray, 127, 255,cv.THRESH_BINARY_INV)  
# print(thresh[0])
# print(ret,thresh)
cv.imshow("image",thresh)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  
cv.drawContours(im, contours,-1,(0,255,0),3)
cv.imshow("im",im)
cv.drawContours(thresh,contours,-1,3)
# cv.imshow("image",contours)
print(len(contours))
cv.waitKey(0)
cv.destroyAllWindows()