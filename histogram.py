import cv2

def histogram(img, mask=None):
    """
    Her kanal veya gri seviye için histogram hesaplar.

    Argümanlar:
        img (numpy.ndarray): Girdi görüntüsü (BGR veya gri).
        mask (numpy.ndarray, optional): Histogramın hesaplanacağı maske.

    Döndürülen:
        dict veya numpy.ndarray: BGR için kanal bazlı histogram sözlüğü veya gri için dizi.
    """
    if len(img.shape) == 2:
        return cv2.calcHist([img], [0], mask, [256], [0,256])
    chans = cv2.split(img)
    hist = {}
    for i, col in enumerate(('b','g','r')):
        hist[col] = cv2.calcHist([chans[i]], [0], mask, [256], [0,256])
    return hist

def equalize_gray(img):
    """
    Gri seviye görüntüye global histogram eşitleme uygular.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.

    Döndürülen:
        numpy.ndarray: Eşitlenmiş gri seviye görüntü.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def clahe_gray(img, clip=2.0, tile=(8,8)):
    """
    Gri seviye görüntüye CLAHE (uyarlanabilir histogram eşitleme) uygular.

    Argümanlar:
        img (numpy.ndarray): Girdi BGR görüntüsü.
        clip (float): Kontrast sınırlama eşiği.
        tile (tuple): Eşitleme için hücre boyutu.

    Döndürülen:
        numpy.ndarray: CLAHE uygulanmış gri seviye görüntü.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(gray)