import cv2 

image = cv2.imread("sim9.png")
dst = cv2.resize(image,(2783,2344))
cv2.imwrite("sim9_.png",dst)