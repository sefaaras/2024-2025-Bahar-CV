import cv2

def global_threshold(img, thresh=127, maxval=255):
    """
    Görüntüde sabit eşik değeri uygular.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.
        thresh (int): Eşik değeri.
        maxval (int): Eşik üstü piksel değeri.

    Döndürülen:
        numpy.ndarray: İkili (binary) görüntü.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, out = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)
    return out


def otsu_threshold(img):
    """
    Otsu yöntemiyle otomatik eşikleme yapar.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.

    Döndürülen:
        numpy.ndarray: İkili (binary) görüntü.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def adaptive_mean(img, block_size=11, C=2):
    """
    Ortalama tabanlı adaptif eşikleme yapar.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.
        block_size (int): Komşuluk boyutu (tek sayı).
        C (int): Sabit çıkarma değeri.

    Döndürülen:
        numpy.ndarray: İkili (binary) görüntü.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)


def adaptive_gaussian(img, block_size=11, C=2):
    """
    Gaussian tabanlı adaptif eşikleme yapar.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.
        block_size (int): Komşuluk boyutu (tek sayı).
        C (int): Sabit çıkarma değeri.

    Döndürülen:
        numpy.ndarray: İkili (binary) görüntü.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)