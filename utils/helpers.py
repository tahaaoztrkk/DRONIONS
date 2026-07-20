"""
Görevi: Ortak yardımcı fonksiyonlar (bbox dönüşümleri, renk, zaman vb.)
"""
import cv2
import time

def get_bbox_center(bbox):
    """Returns (x_center, y_center) of a bbox (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_bbox_area(bbox):
    """Returns area of a bbox (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

def get_bbox_aspect_ratio(bbox):
    """Returns width / height of a bbox."""
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return w / h

def time_it(func):
    """Decorator to measure execution time."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {(end-start)*1000:.2f} ms")
        return result
    return wrapper
