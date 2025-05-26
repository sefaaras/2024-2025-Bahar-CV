import cv2
import numpy as np


def harris_corners(img, block_size=2, ksize=3, k=0.04):
    """
    Harris algoritmasıyla köşe tespiti yapar.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR veya gri görüntü.
        block_size (int): Pencere boyutu.
        ksize (int): Sobel çekirdek boyutu.
        k (float): Harris parametresi.

    Döndürülen:
        numpy.ndarray: Köşe haritası (float değerler).
    """
    gray = img if len(img.shape)==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    return cv2.cornerHarris(gray, block_size, ksize, k)


def orb_features(img, n_features=500):
    """
    ORB detektörü kullanarak anahtar nokta ve öznitelikleri çıkarır.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.
        n_features (int): Maksimum öznitelik sayısı.

    Döndürülen:
        keypoints, descriptors
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(n_features)
    return orb.detectAndCompute(gray, None)

