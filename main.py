import os
import cv2
import numpy as np
import inout, color, histogram, filtering, morphology, edge, thresholding, segmentation, features, transform

# Örnek görüntü yolu
TEST_IMAGE = 'lena.png'  # Test için uygun bir görüntü yerleştirin

# Çıktı dizini oluştur
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Görüntüyü oku
img = inout.read_image(TEST_IMAGE)
if img is None:
    raise FileNotFoundError(f"Görüntü bulunamadı: {TEST_IMAGE}")

# 1) IO test: göster ve kaydet
inout.save_image(os.path.join(OUTPUT_DIR, 'io_saved.jpg'), img)
# io.show_image('Original', img)

# 2) Color dönüşümleri
gray = color.to_grayscale(img)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'gray.jpg'), gray)

hsv = color.to_hsv(img)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'hsv.jpg'), hsv)

# 3) Histogram
hist = histogram.histogram(img)
# Kanal bazlı histogramları birleştirip göster (opsiyonel)

# 4) Filtreleme
cv2.imwrite(os.path.join(OUTPUT_DIR, 'gaussian.jpg'), filtering.gaussian_blur(img))
cv2.imwrite(os.path.join(OUTPUT_DIR, 'median.jpg'), filtering.median_blur(img))
cv2.imwrite(os.path.join(OUTPUT_DIR, 'bilateral.jpg'), filtering.bilateral_filter(img))

# 5) Morfoloji (örnek binary üzerinden)
bin_img = thresholding.global_threshold(img)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'eroded.jpg'), morphology.erode(bin_img))
cv2.imwrite(os.path.join(OUTPUT_DIR, 'dilated.jpg'), morphology.dilate(bin_img))

# 6) Kenar algılama
cv2.imwrite(os.path.join(OUTPUT_DIR, 'sobel.jpg'), edge.sobel_edges(img))
cv2.imwrite(os.path.join(OUTPUT_DIR, 'canny.jpg'), edge.canny_edges(img))

# 7) Eşikleme
cv2.imwrite(os.path.join(OUTPUT_DIR, 'otsu.jpg'), thresholding.otsu_threshold(img))

# 8) Segmentasyon
cv2.imwrite(os.path.join(OUTPUT_DIR, 'kmeans_seg.jpg'), segmentation.kmeans_segmentation(img))
# SLIC segment görüntüyü renk haritası olarak kaydet
slic_mask = segmentation.slic_superpixels(img)
# Normalizasyon
cv2.imwrite(os.path.join(OUTPUT_DIR, 'slic.png'), (slic_mask / slic_mask.max() * 255).astype(np.uint8))

# 9) Özellik çıkarma
corners = features.harris_corners(img)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'harris.jpg'), (corners > 0.01 * corners.max()).astype(np.uint8) * 255)
kp, des = features.orb_features(img)
out = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
cv2.imwrite(os.path.join(OUTPUT_DIR, 'orb.jpg'), out)

# 10) Dönüşümler
fourier = transform.fourier_transform(img)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'fourier.jpg'), fourier.astype(np.uint8))

sinogram = transform.radon_transform(img)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'radon.jpg'), (sinogram / sinogram.max() * 255).astype(np.uint8))

print('Tüm modüller test edildi. Çıktılar', OUTPUT_DIR, 'klasöründe.')