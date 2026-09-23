import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# MEMBACA CITRA
# =========================================================

im = cv2.imread("gambarcontoh.webp", cv2.IMREAD_GRAYSCALE)

if im is None:
    print("Gambar tidak ditemukan!")
    exit()

print("Ukuran citra:", im.shape)


# =========================================================
# LOWPASS FILTER
# =========================================================

# ---------- Box Filter ----------
kernel_box = np.ones((3, 3), np.float32) / 9.0

hasil_box = cv2.filter2D(
    im,
    -1,
    kernel_box,
    borderType=cv2.BORDER_REPLICATE
)

# Cara singkat Box Filter
hasil_box2 = cv2.blur(im, (3, 3))


# ---------- Gaussian Filter ----------
kernel_gauss = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], np.float32) / 16.0

hasil_gauss = cv2.filter2D(
    im,
    -1,
    kernel_gauss,
    borderType=cv2.BORDER_REPLICATE
)

# Cara singkat Gaussian Filter
hasil_gauss2 = cv2.GaussianBlur(
    im,
    (5, 5),
    sigmaX=1.0
)


# ---------- Median Filter ----------
hasil_median = cv2.medianBlur(im, 3)


# =========================================================
# HIGHPASS FILTER
# =========================================================

imf = im.astype(np.float64)


# ---------- Laplacian ----------
kernel_lap = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], np.float64)

lap = cv2.filter2D(
    imf,
    -1,
    kernel_lap,
    borderType=cv2.BORDER_REPLICATE
)

# Karena pusat kernel = -4,
# citra dipertajam dengan mengurangi Laplacian
tajam = imf - lap

# Batasi nilai ke rentang 0-255
tajam = np.clip(
    tajam,
    0,
    255
).astype(np.uint8)


# Laplacian bawaan OpenCV
lap_cv = cv2.Laplacian(
    im,
    cv2.CV_64F,
    ksize=1
)


# =========================================================
# UNSHARP MASKING DAN HIGHBOOST
# =========================================================

def unsharp(im, k=1.0, ukuran=3):

    imf = im.astype(np.float64)

    # Membuat citra halus
    halus = cv2.blur(
        imf,
        (ukuran, ukuran)
    )

    # Membuat mask
    mask = imf - halus

    # Penajaman
    hasil = imf + k * mask

    # Batasi rentang 0-255
    return np.clip(
        hasil,
        0,
        255
    ).astype(np.uint8)


# Unsharp Masking
hasil_unsharp = unsharp(
    im,
    k=1.0
)


# Highboost Filtering
hasil_highboost = unsharp(
    im,
    k=2.5
)


# =========================================================
# SOBEL
# =========================================================

# Sobel arah X
gx = cv2.Sobel(
    imf,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

# Sobel arah Y
gy = cv2.Sobel(
    imf,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)


# Besar gradien
besar = np.abs(gx) + np.abs(gy)


# Besar gradien sebenarnya
besar_tepat = np.sqrt(
    gx ** 2 + gy ** 2
)


# Arah gradien
arah = np.arctan2(
    gy,
    gx
)


# Membuat citra tepi
tepi = np.clip(
    besar,
    0,
    255
).astype(np.uint8)


# =========================================================
# MENAMPILKAN HASIL DENGAN MATPLOTLIB
# =========================================================

plt.figure(figsize=(14, 10))


# Original
plt.subplot(3, 3, 1)
plt.imshow(im, cmap="gray")
plt.title("Original")
plt.axis("off")


# Box
plt.subplot(3, 3, 2)
plt.imshow(hasil_box, cmap="gray")
plt.title("Box Filter")
plt.axis("off")


# Gaussian
plt.subplot(3, 3, 3)
plt.imshow(hasil_gauss, cmap="gray")
plt.title("Gaussian Filter")
plt.axis("off")


# Median
plt.subplot(3, 3, 4)
plt.imshow(hasil_median, cmap="gray")
plt.title("Median Filter")
plt.axis("off")


# Laplacian
plt.subplot(3, 3, 5)
plt.imshow(tajam, cmap="gray")
plt.title("Laplacian Sharpening")
plt.axis("off")


# Unsharp
plt.subplot(3, 3, 6)
plt.imshow(hasil_unsharp, cmap="gray")
plt.title("Unsharp Masking")
plt.axis("off")


# Highboost
plt.subplot(3, 3, 7)
plt.imshow(hasil_highboost, cmap="gray")
plt.title("Highboost")
plt.axis("off")


# Sobel
plt.subplot(3, 3, 8)
plt.imshow(tepi, cmap="gray")
plt.title("Sobel Edge")
plt.axis("off")


plt.tight_layout()
plt.show()