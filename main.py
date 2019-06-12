import cv2
import keras
import numpy as np
import sys
import random as rng
from matplotlib import pyplot as plt

# Prvi i jedini argument komandne linije je indeks test primera
if len(sys.argv) != 2:
    print("Neispravno pozvan fajl, koristiti komandu \"python3 main.py X\" za pokretanje na test primeru sa brojem X")
    exit(0)

tp_idx = sys.argv[1]
img = cv2.imread('tests/{}.png'.format(tp_idx))

#################################################################################
# U ovoj sekciji implementirati obradu slike, ucitati prethodno trenirani Keras
# model, i dodati bounding box-ove i imena klasa na sliku.
# Ne menjati fajl van ove sekcije.

# Ucitavamo model
# model = keras.models.load_model('fashion.h5')

solution = img.copy()

dst = cv2.fastNlMeansDenoising(solution,None,50,7,21)
# plt.subplot(121),plt.imshow(solution)
# plt.subplot(122),plt.imshow(dst)
# plt.show()
src_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
src_gray = cv2.blur(src_gray, (3,3))

def thresh_callback(val):
    threshold = val

    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
                     (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
        # cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

    cv2.imshow('Contours', drawing)

source_window = 'Source'
cv2.namedWindow(source_window)
cv2.imshow(source_window, img)

max_thresh = 255
thresh = 10 # initial threshold
cv2.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)

# black_pixels = np.where(dst <= 240)
# coords = np.column_stack((black_pixels[1], black_pixels[0]))
# rect = cv2.minAreaRect(coords)
# box = np.int0(np.around(cv2.boxPoints(rect)))
# cv2.drawContours(solution, [box],0,255,5)
# cv2.imshow('', solution)
cv2.waitKey(0)

#################################################################################

# Cuvamo resenje u izlazni fajl
# cv2.imwrite("tests/{}_out.png".format(tp_idx), solution)