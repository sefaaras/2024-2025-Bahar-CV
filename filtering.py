import cv2
import numpy as np
from skimage.restoration import denoise_bilateral

def gaussian_blur(img, ksize=(5,5), sigma=1.0):
    """
    Gaussian çekirdek ile görüntüyü bulanıklaştırır.

    Argümanlar:
        img (numpy.ndarray): Girdi görüntüsü.
        ksize (tuple): Çekirdek boyutu.
        sigma (float): Gaussian standart sapması.

    Döndürülen:
        numpy.ndarray: Bulanıklaştırılmış görüntü.
    """
    return cv2.GaussianBlur(img, ksize, sigma)

def median_blur(img, ksize=5):
    """
    Medyan filtre ile gürültü azaltır.

    Argümanlar:
        img (numpy.ndarray): Girdi görüntüsü.
        ksize (int): Çekirdek boyutu (tek sayı).

    Döndürülen:
        numpy.ndarray: Gürültü azaltılmış görüntü.
    """
    return cv2.medianBlur(img, ksize)

def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    """
    Kenarları koruyarak görüntüyü süzer.

    Argümanlar:
        img (numpy.ndarray): Girdi görüntüsü.
        d (int): Komşuluk çapı.
        sigma_color (float): Renk uzayında sigma.
        sigma_space (float): Uzay uzayında sigma.

    Döndürülen:
        numpy.ndarray: Kenar korumalı süzülmüş görüntü.
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def skimage_bilateral(img, sigma_color=0.05, sigma_spatial=15):
    """
    scikit-image ile bilateral gürültü giderme yapar.

    Argümanlar:
        img (numpy.ndarray): Girdi görüntüsü.
        sigma_color (float): Renk sigması (0-1 arası).
        sigma_spatial (int): Uzay sigması.

    Döndürülen:
        numpy.ndarray: Gürültü giderilmiş görüntü.
    """
    img_f = img.astype(np.float32) / 255.0
    denoised = denoise_bilateral(img_f, sigma_color=sigma_color, sigma_spatial=sigma_spatial, multichannel=True)
    return (denoised * 255).astype(np.uint8)