import cv2

def transform_img(img, width, height):
	img = cv2.resize(img,(width,height), interpolation= cv2.INTER_CUBIC)
	