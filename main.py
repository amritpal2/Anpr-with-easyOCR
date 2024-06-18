import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

img = cv2.imread('image1.jpg')

if img is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
plt.title("Grayscale Image")
plt.show()

bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

edged = cv2.Canny(bfilter, 30, 200)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB))
plt.title("Edge Detection")
plt.show()

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

if location is None:
    raise ValueError("No rectangular contour found.")

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title("Masked Image")
plt.show()

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB))
plt.title("Cropped Image")
plt.show()

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
# result1 = reader.readtext(new_image)


if len(result) == 0:
    raise ValueError("No text detected.")

text = result[0][-2]

font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(location[1][0][0], location[0][0][1]+60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.title("Final Image with Detected Text")
plt.show()
