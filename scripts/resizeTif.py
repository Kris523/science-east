import cv2
import sys

input = sys.argv[1]
output_path = sys.argv[2]

img = cv2.imread(input, cv2.IMREAD_UNCHANGED)

oldY, oldX, depth = img.shape
desiredX = 3267
scaleRatio = desiredX/oldX
print(scaleRatio)

newimg = cv2.resize(img,(int(oldX*scaleRatio), int(oldY*scaleRatio)))


cv2.imwrite(output_path + '/old.png', img)
cv2.imwrite(output_path + '/2013.png', newimg)
cv2.imwrite(output_path + 'old2.png', img)

# cv2.imshow("img", newimg)
# cv2.waitKey()
print("complete")
