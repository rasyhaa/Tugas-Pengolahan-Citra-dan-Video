import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. TRANSFORMASI NEGATIF
# ============================================================

def negative(image):

    height, width = image.shape

    result = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):

            r = int(image[y, x])

            # Rumus negatif
            s = 255 - r

            result[y, x] = s

    return result


# ============================================================
# 2. TRANSFORMASI LOGARITMIK
# ============================================================

def log_transform(image):

    height, width = image.shape

    result = np.zeros((height, width), dtype=np.uint8)

    # Cari nilai maksimum secara manual
    max_value = 0

    for y in range(height):
        for x in range(width):

            value = int(image[y, x])

            if value > max_value:
                max_value = value

    # Konstanta c
    c = 255.0 / np.log(1 + max_value)

    # Hitung setiap pixel satu per satu
    for y in range(height):
        for x in range(width):

            r = int(image[y, x])

            s = c * np.log(1 + r)

            if s < 0:
                s = 0

            if s > 255:
                s = 255

            result[y, x] = int(s)

    return result


# ============================================================
# 3. TRANSFORMASI POWER LAW / GAMMA
# ============================================================

def gamma_transform(image, gamma):

    height, width = image.shape

    result = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):

            r = int(image[y, x])

            # Normalisasi 0 - 1
            normalized = r / 255.0

            # Rumus gamma
            s = normalized ** gamma

            # Kembalikan ke 0 - 255
            s = s * 255.0

            if s < 0:
                s = 0

            if s > 255:
                s = 255

            result[y, x] = int(s)

    return result


# ============================================================
# 4. CONTRAST STRETCHING
# ============================================================

def contrast_stretching(image):

    height, width = image.shape

    result = np.zeros((height, width), dtype=np.uint8)

    # --------------------------------------------------------
    # Mencari nilai minimum dan maksimum secara manual
    # --------------------------------------------------------

    min_value = 255
    max_value = 0

    for y in range(height):
        for x in range(width):

            value = int(image[y, x])

            if value < min_value:
                min_value = value

            if value > max_value:
                max_value = value

    # --------------------------------------------------------
    # Transformasi setiap pixel
    # --------------------------------------------------------

    for y in range(height):
        for x in range(width):

            r = int(image[y, x])

            if max_value == min_value:

                s = r

            else:

                s = ((r - min_value) /
                     (max_value - min_value)) * 255

            if s < 0:
                s = 0

            if s > 255:
                s = 255

            result[y, x] = int(s)

    return result


# ============================================================
# 5. HISTOGRAM MANUAL
# ============================================================

def calculate_histogram(image):

    height, width = image.shape

    # 256 tingkat intensitas
    histogram = np.zeros(256, dtype=np.int64)

    # Hitung pixel satu per satu
    for y in range(height):
        for x in range(width):

            intensity = int(image[y, x])

            histogram[intensity] = histogram[intensity] + 1

    return histogram


# ============================================================
# 6. HISTOGRAM EQUALIZATION MANUAL
# ============================================================

def histogram_equalization(image):

    height, width = image.shape

    # --------------------------------------------------------
    # LANGKAH 1: Histogram
    # --------------------------------------------------------

    histogram = calculate_histogram(image)

    # --------------------------------------------------------
    # LANGKAH 2: Total pixel
    # --------------------------------------------------------

    total_pixels = height * width

    # --------------------------------------------------------
    # LANGKAH 3: CDF
    # --------------------------------------------------------

    cdf = np.zeros(256, dtype=np.int64)

    cdf[0] = histogram[0]

    for i in range(1, 256):

        cdf[i] = cdf[i - 1] + histogram[i]

    # --------------------------------------------------------
    # LANGKAH 4: Cari CDF minimum
    # --------------------------------------------------------

    cdf_min = 0

    for i in range(256):

        if cdf[i] > 0:

            cdf_min = cdf[i]

            break

    # --------------------------------------------------------
    # LANGKAH 5: Mapping intensitas
    # --------------------------------------------------------

    mapping = np.zeros(256, dtype=np.uint8)

    for i in range(256):

        if total_pixels == cdf_min:

            mapping[i] = 0

        else:

            value = (
                (cdf[i] - cdf_min) /
                (total_pixels - cdf_min)
            ) * 255

            if value < 0:
                value = 0

            if value > 255:
                value = 255

            mapping[i] = int(value)

    # --------------------------------------------------------
    # LANGKAH 6: Terapkan mapping
    # --------------------------------------------------------

    result = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):

            old_value = int(image[y, x])

            new_value = mapping[old_value]

            result[y, x] = new_value

    return result


# ============================================================
# 7. PROGRAM UTAMA
# ============================================================

# Membaca gambar
image = cv2.imread("gambarcontoh.webp")


# Cek gambar
if image is None:

    print("Gambar tidak ditemukan!")

    exit()


# Konversi ke grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ============================================================
# TRANSFORMASI INTENSITAS
# ============================================================

negative_image = negative(gray)

log_image = log_transform(gray)

gamma_image = gamma_transform(gray, 0.5)

contrast_image = contrast_stretching(gray)


# ============================================================
# HISTOGRAM EQUALIZATION
# ============================================================

equalized_image = histogram_equalization(gray)


# ============================================================
# HISTOGRAM
# ============================================================

hist_original = calculate_histogram(gray)

hist_equalized = calculate_histogram(equalized_image)


# ============================================================
# MENAMPILKAN HASIL
# ============================================================

plt.figure(figsize=(15, 10))


plt.subplot(2, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")


plt.subplot(2, 3, 2)
plt.imshow(negative_image, cmap="gray")
plt.title("Negative")
plt.axis("off")


plt.subplot(2, 3, 3)
plt.imshow(log_image, cmap="gray")
plt.title("Log Transform")
plt.axis("off")


plt.subplot(2, 3, 4)
plt.imshow(gamma_image, cmap="gray")
plt.title("Gamma Transform")
plt.axis("off")


plt.subplot(2, 3, 5)
plt.imshow(contrast_image, cmap="gray")
plt.title("Contrast Stretching")
plt.axis("off")


plt.subplot(2, 3, 6)
plt.imshow(equalized_image, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# MENAMPILKAN HISTOGRAM
# ============================================================

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)

plt.bar(
    range(256),
    hist_original,
    width=1
)

plt.title("Histogram Original")
plt.xlabel("Intensitas")
plt.ylabel("Jumlah Pixel")


plt.subplot(1, 2, 2)

plt.bar(
    range(256),
    hist_equalized,
    width=1
)

plt.title("Histogram Equalization")
plt.xlabel("Intensitas")
plt.ylabel("Jumlah Pixel")


plt.tight_layout()
plt.show()