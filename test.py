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

corners = features.harris_corners(img)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'harris.jpg'), (corners > 0.01 * corners.max()).astype(np.uint8) * 255)
kp, des = features.orb_features(img)
out = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
cv2.imwrite(os.path.join(OUTPUT_DIR, 'orb.jpg'), out)

