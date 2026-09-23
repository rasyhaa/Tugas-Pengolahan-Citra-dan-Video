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

# Filter warna
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


# FILTER COLOR WEBCAM

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Kamera tidak dapat dibuka!")
        break

    # FILTER RGB

    # Original
    original = frame.copy()

    # Red
    red_filter = np.zeros_like(frame)
    red_filter[:, :, 2] = frame[:, :, 2]

    # Green
    green_filter = np.zeros_like(frame)
    green_filter[:, :, 1] = frame[:, :, 1]

    # Blue
    blue_filter = np.zeros_like(frame)
    blue_filter[:, :, 0] = frame[:, :, 0]

    # GABUNGKAN SEMUA FILTER

    # Ukuran frame
    height, width = frame.shape[:2]

    # Gabungkan horizontal:
    # Original | Red
    atas = np.hstack((original, red_filter))

    # Green | Blue
    bawah = np.hstack((green_filter, blue_filter))

    # Gabungkan vertikal
    hasil = np.vstack((atas, bawah))


    # LABEL

    cv2.putText(
        hasil,
        "ORIGINAL",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        hasil,
        "RED",
        (width + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        hasil,
        "GREEN",
        (20, height + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        hasil,
        "BLUE",
        (width + 20, height + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )


    # SHOW
    cv2.imshow("Webcam RGB Filter", hasil)


    # ESC untuk keluar
    if cv2.waitKey(1) == 27:
        break

# RELEASE

cap.release()
cv2.destroyAllWindows()