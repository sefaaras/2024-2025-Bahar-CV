import cv2
import numpy as np
from skimage.transform import radon

def fourier_transform(img):
    """
    Görüntünün Fourier dönüşümünü ve spektrumunu hesaplar.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR veya gri görüntü.

    Döndürülen:
        magnitude_spectrum (numpy.ndarray)
    """
    gray = img if len(img.shape)==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum


def radon_transform(img, theta=None):
    """
    Radon dönüşümü uygular.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR veya gri görüntü.
        theta (array-like, optional): Açı dizisi derece cinsinden.

    Döndürülen:
        numpy.ndarray: Radon dönüşüm matrisi.
    """
    gray = img if len(img.shape)==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if theta is None:
        theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(gray, theta=theta, circle=True)
    return sinogram


def hough_lines(img, rho=1, theta=np.pi/180, threshold=100):
    """
    Hough dönüşümü ile çizgi tespiti yapar.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.
        rho (float): Mesafe çözünürlüğü.
        theta (float): Açı çözünürlüğü.
        threshold (int): Oyuğun tespit eşiği.

    Döndürülen:
        lines: Tespit edilen çizgiler.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    return lines


def hough_circles(img, dp=1.2, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0):
    """
    Hough dönüşümü ile daire tespiti yapar.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.
        dp (float): Birikmiş imge çözünürlüğü oranı.
        minDist (float): Daire merkezleri arası minimum mesafe.
        param1 (float): Canny üst eşiği.
        param2 (float): Merkez tespiti eşik değeri.
        minRadius (int): Minimum yarıçap.
        maxRadius (int): Maksimum yarıçap.

    Döndürülen:
        circles: Tespit edilen daire parametreleri.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    return circles