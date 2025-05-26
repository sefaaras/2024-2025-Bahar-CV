import cv2
import numpy as np

def sobel_edges(img):
    """Sobel ile x ve y kenarları."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(dx, dy)
    return np.uint8(np.clip(mag, 0, 255))

def canny_edges(img, th1=100, th2=200):
    """Canny kenar algılama."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, th1, th2)
