import cv2
import numpy as np
from skimage.segmentation import slic
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def kmeans_segmentation(img, k=3):
    """
    Renk uzayında K-means algoritmasıyla segmentasyon yapar.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.
        k (int): Segment sayısı.

    Döndürülen:
        numpy.ndarray: Segmentlenmiş görüntü.
    """
    Z = img.reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img.shape)
    return segmented


def slic_superpixels(img, n_segments=5, compactness=2):
    """
    SLIC algoritmasıyla süper-piksel segmentasyonu yapar.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.
        n_segments (int): Oluşturulacak süper-piksel sayısı.
        compactness (float): Piksel uyumluluk ağırlığı.

    Döndürülen:
        numpy.ndarray: Renkli süper-piksel haritası.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    segments = slic(img_rgb, n_segments=n_segments, compactness=compactness)
    return segments


def watershed_segmentation(img):
    """
    Watershed algoritmasıyla segmentasyon yapar.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.

    Döndürülen:
        numpy.ndarray: Etiketlenmiş segment haritası.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gürültü azaltma
    denoised = cv2.GaussianBlur(gray, (3,3), 0)
    # İkili maske
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # uzaklık dönüşümü
    distance = ndi.distance_transform_edt(thresh)
    # zirve noktaları
    local_max = peak_local_max(distance, indices=False, footprint=np.ones((3,3)))
    markers = ndi.label(local_max)[0]
    labels = watershed(-distance, markers, mask=thresh)
    return labels