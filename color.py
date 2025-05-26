import cv2

def to_grayscale(img):
    """
    BGR görüntüyü gri seviye görüntüye çevirir.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.

    Döndürülen:
        numpy.ndarray: Tek kanal gri seviye görüntü.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_hsv(img):
    """
    BGR görüntüyü HSV renk uzayına dönüştürür.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.

    Döndürülen:
        numpy.ndarray: HSV renk uzayındaki görüntü.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def to_lab(img):
    """
    BGR görüntüyü LAB renk uzayına dönüştürür.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.

    Döndürülen:
        numpy.ndarray: LAB renk uzayındaki görüntü.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

def to_ycrcb(img):
    """
    BGR görüntüyü YCrCb renk uzayına dönüştürür.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.

    Döndürülen:
        numpy.ndarray: YCrCb renk uzayındaki görüntü.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

def hsv_to_bgr(img):
    """
    HSV görüntüyü tekrar BGR formatına çevirir.

    Argümanlar:
        img (numpy.ndarray): Girdi HSV görüntüsü.

    Döndürülen:
        numpy.ndarray: BGR formatındaki görüntü.
    """
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def lab_to_bgr(img):
    """
    LAB görüntüyü tekrar BGR formatına çevirir.

    Argümanlar:
        img (numpy.ndarray): Girdi LAB görüntüsü.

    Döndürülen:
        numpy.ndarray: BGR formatındaki görüntü.
    """
    return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

def ycrcb_to_bgr(img):
    """
    YCrCb görüntüyü tekrar BGR formatına çevirir.

    Argümanlar:
        img (numpy.ndarray): Girdi YCrCb görüntüsü.

    Döndürülen:
        numpy.ndarray: BGR formatındaki görüntü.
    """
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)