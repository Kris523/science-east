import cv2

img = cv2.imread('/home/eigen/2003.jpg', cv2.IMREAD_UNCHANGED)

oldY, oldX, depth = img.shape
desiredX = 3267
scaleRatio = desiredX/oldX
print(scaleRatio)

newimg = cv2.resize(img,(int(oldX*scaleRatio), int(oldY*scaleRatio)))


cv2.imwrite('/home/eigen/Pictures/old.png', img)
cv2.imwrite('/home/eigen/Pictures/2013.png', newimg)
cv2.imwrite('/home/eigen/Pictures/old2.png', img)

# cv2.imshow("img", newimg)
# cv2.waitKey()
print("complete")