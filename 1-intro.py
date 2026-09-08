import cv2
import numpy as np


# READ IMAGE
img = cv2.imread("gambarcontoh.webp")

# SHOW IMAGE
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# FILTER COLOR IMAGE
red = np.zeros_like(img)
green = np.zeros_like(img)
blue = np.zeros_like(img)

# Filter
red[:, :, 2] = img[:, :, 2]
green[:, :, 1] = img[:, :, 1]
blue[:, :, 0] = img[:, :, 0]

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Hasil
cv2.imshow("Red", red)
cv2.imshow("Green", green)
cv2.imshow("Blue", blue)
cv2.imshow("Grayscale", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

# FILTER COLOR VIDEO
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Membuat gambar
    filter = np.zeros_like(frame)

    # Filter 0 = biru, 1 = hijau, 2 = merah
    filter[:, :, 2] = frame[:, :, 2]

    # Hasil
    cv2.imshow("Filter", filter)

    # Tekan ESC untuk keluar
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()