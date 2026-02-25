import cv2
import numpy as np

def enhance_book_image(path):
    img = cv2.imread(path)

    # resize large images
    h, w = img.shape[:2]
    scale = 900 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # increase contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    img = cv2.merge((l,a,b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    # sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img,-1,kernel)

    cv2.imwrite(path, img)