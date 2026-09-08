import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Membuat gambar
    filter = np.zeros_like(frame)

    # Membuat filter 0(biru),1(hijau),2(merah)
    filter[:, :, 2] = frame[:, :, 2]

    # filter
    cv2.imshow("Filter", filter)

    # Tekan ESC untuk keluar
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()