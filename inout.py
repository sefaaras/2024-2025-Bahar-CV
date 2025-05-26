import cv2
import os

def read_image(path, flags=cv2.IMREAD_COLOR):
    """
    Diskten görüntü okur.

    Argümanlar:
        path (str): Görüntü dosya yolu.
        flags (int, optional): OpenCV okuma bayrakları (örn. cv2.IMREAD_COLOR). Varsayılan cv2.IMREAD_COLOR.

    Döndürülen:
        numpy.ndarray: Yüklenen görüntü dizisi veya yükleme başarısızsa None.
    """
    return cv2.imread(path, flags)


def save_image(path, img):
    """
    Görüntüyü diske kaydeder, gerekirse klasör oluşturur.

    Argümanlar:
        path (str): Kaydedilecek dosya yolu.
        img (numpy.ndarray): Kaydedilecek görüntü verisi.

    Döndürülen:
        bool: Kaydetme başarılıysa True, aksi halde False.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return cv2.imwrite(path, img)


def show_image(window_name, img, wait=True):
    """
    Görüntüyü ekranda pencerede gösterir.

    Argümanlar:
        window_name (str): Pencere adı.
        img (numpy.ndarray): Gösterilecek görüntü.
        wait (bool, optional): True ise tuşa basılana kadar bekler. Varsayılan True.
    """
    cv2.imshow(window_name, img)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def read_video(path):
    """
    Video dosyasını okumak için açar.

    Argümanlar:
        path (str): Video dosya yolu.

    Döndürülen:
        cv2.VideoCapture: Video yakalama nesnesi.
    """
    return cv2.VideoCapture(path)


def write_video(path, fourcc, fps, frame_size):
    """
    Video karelerini kaydetmek için VideoWriter oluşturur.

    Argümanlar:
        path (str): Çıktı video dosya yolu.
        fourcc (int): Codec FourCC kodu (örn. cv2.VideoWriter_fourcc(*'XVID')).
        fps (float): Kare hızı.
        frame_size (tuple): Kare boyutu (genişlik, yükseklik).

    Döndürülen:
        cv2.VideoWriter: Video yazma nesnesi.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return cv2.VideoWriter(path, fourcc, fps, frame_size)