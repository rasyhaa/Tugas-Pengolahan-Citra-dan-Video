import cv2
import numpy as np

img = cv2.imread("wp3997619.webp")

# Membuat gambar
red = np.zeros_like(img)
green = np.zeros_like(img)
blue = np.zeros_like(img)

# Filter
red[:, :, 2] = img[:, :, 2]
green[:, :, 1] = img[:, :, 1]
blue[:, :, 0] = img[:, :, 0]

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# hasil
cv2.imshow("Red", red)
cv2.imshow("Green", green)
cv2.imshow("Blue", blue)
cv2.imshow("Grayscale", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()