import cv2

def erode(img, kernel=None, iterations=1):
    """
    Erozyon işlemi: görüntüdeki parlak bölgeleri küçültür.

    Argümanlar:
        img (numpy.ndarray): Girdi ikili veya gri görüntü.
        kernel (numpy.ndarray, optional): Morfolojik çekirdek.
        iterations (int): Erozyon tekrarı sayısı.

    Döndürülen:
        numpy.ndarray: Erozyon uygulanmış görüntü.
    """
    k = kernel if kernel is not None else cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.erode(img, k, iterations=iterations)


def dilate(img, kernel=None, iterations=1):
    """
    Dilation işlemi: görüntüdeki parlak bölgeleri genişletir.

    Argümanlar:
        img (numpy.ndarray): Girdi ikili veya gri görüntü.
        kernel (numpy.ndarray, optional): Morfolojik çekirdek.
        iterations (int): Dilasyon tekrarı sayısı.

    Döndürülen:
        numpy.ndarray: Dilasyon uygulanmış görüntü.
    """
    k = kernel if kernel is not None else cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.dilate(img, k, iterations=iterations)


def opening(img, kernel=None):
    """
    Opening işlemi: önce erozyon, sonra dilasyon (gürültü giderme).

    Argümanlar:
        img (numpy.ndarray): Girdi görüntü.
        kernel (numpy.ndarray, optional): Morfolojik çekirdek.

    Döndürülen:
        numpy.ndarray: Opening uygulanmış görüntü.
    """
    k = kernel if kernel is not None else cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, k)


def closing(img, kernel=None):
    """
    Closing işlemi: önce dilasyon, sonra erozyon (küçük delikleri kapatma).

    Argümanlar:
        img (numpy.ndarray): Girdi görüntü.
        kernel (numpy.ndarray, optional): Morfolojik çekirdek.

    Döndürülen:
        numpy.ndarray: Closing uygulanmış görüntü.
    """
    k = kernel if kernel is not None else cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)


def gradient(img, kernel=None):
    """
    Morfolojik gradyan: dilasyon ile erozyon farkı.

    Argümanlar:
        img (numpy.ndarray): Girdi görüntü.
        kernel (numpy.ndarray, optional): Morfolojik çekirdek.

    Döndürülen:
        numpy.ndarray: Gradyan görüntü.
    """
    k = kernel if kernel is not None else cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)


def top_hat(img, kernel=None):
    """
    Top-hat dönüşümü: orijinal görüntü minus opening.

    Argümanlar:
        img (numpy.ndarray): Girdi görüntü.
        kernel (numpy.ndarray, optional): Morfolojik çekirdek.

    Döndürülen:
        numpy.ndarray: Top-hat sonucu.
    """
    k = kernel if kernel is not None else cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k)


def black_hat(img, kernel=None):
    """
    Black-hat dönüşümü: closing minus orijinal görüntü.

    Argümanlar:
        img (numpy.ndarray): Girdi görüntü.
        kernel (numpy.ndarray, optional): Morfolojik çekirdek.

    Döndürülen:
        numpy.ndarray: Black-hat sonucu.
    """
    k = kernel if kernel is not None else cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k)