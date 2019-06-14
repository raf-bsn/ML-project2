import cv2
import keras
import numpy as np
import sys
from copy import deepcopy
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
model = keras.models.load_model('fashion2.h5')

solution = img.copy()

dst = cv2.fastNlMeansDenoising(solution,None,50,7,21)

# plt.subplot(121),plt.imshow(solution)
# plt.subplot(122),plt.imshow(dst)
# plt.show()

src_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
src_gray = cv2.blur(src_gray, (3,3))

# dst = deepcopy(src_gray)
# plt.subplot(122),plt.imshow(dst)
# plt.show()

ret, src_gray = cv2.threshold(src_gray, 240, 255, cv2.THRESH_BINARY)


def find_if_close(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 5:
                return True
            elif i == row1-1 and j == row2-1:
                return False


def thresh_callback(val):
    threshold = val

    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))

    for i, cnt1 in enumerate(contours):
        x = i
        if i != LENGTH - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x = x + 1
                dist = find_if_close(cnt1, cnt2)
                if dist is True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i + 1

    unified = []
    maximum = int(status.max()) + 1
    for i in range(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)

    contours = unified

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    for i in range(len(contours)):
        color = (255, 0, 0)
        latitude = boundRect[i][2] * boundRect[i][3]
        if latitude < 100:                                      # izbacivanje malih kontura
            continue
        # cv2.drawContours(img, contours_poly, i, color)
        cv2.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])), \
                     (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)

    return boundRect

# source_window = 'Source'
# cv2.namedWindow(source_window)

max_thresh = 255
thresh = 10 # initial threshold
# cv2.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
boundingBoxes = thresh_callback(thresh)

labelNames = ["tshirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

print(boundingBoxes)

for i in range(len(boundingBoxes)):
    imageHeight = boundingBoxes[i][3]
    imageWidth = boundingBoxes[i][2]
    if imageHeight <= 10 or imageWidth <= 10:
        continue

    if imageWidth > imageHeight:
        boundingBoxes[i] = (boundingBoxes[i][0], boundingBoxes[i][1] - (imageWidth - imageHeight) // 2, boundingBoxes[i][2], boundingBoxes[i][3])
        firstImage = solution[boundingBoxes[i][1]: boundingBoxes[i][1] + boundingBoxes[i][2],
                     boundingBoxes[i][0]: boundingBoxes[i][0] + boundingBoxes[i][2]]
        res = cv2.resize(firstImage, None, fx=28 / imageWidth, fy=28 / imageWidth, interpolation=cv2.INTER_AREA)
    else:
        boundingBoxes[i] = (boundingBoxes[i][0] - (imageHeight - imageWidth) // 2, boundingBoxes[i][1], boundingBoxes[i][2], boundingBoxes[i][3])
        firstImage = solution[boundingBoxes[i][1]: boundingBoxes[i][1] + boundingBoxes[i][3],
                 boundingBoxes[i][0] : boundingBoxes[i][0] + boundingBoxes[i][3]]
        res = cv2.resize(firstImage, None, fx=28 / imageHeight, fy=28 / imageHeight, interpolation=cv2.INTER_AREA)

    # res = np.flip(res, -1)                      # flip vertical, then horizontal
    res = cv2.bitwise_not(res)

    # normalization
    # res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    res = res.astype('float32')
    res = res / 255
    res = cv2.resize(res, (28, 28))

    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res = res.reshape(1, 28, 28, 1)

    probabilities = model.predict(res)
    print(probabilities)
    prediction = probabilities.argmax(axis=1)
    label = labelNames[prediction[0]]
    print(label)
    font = cv2.FONT_HERSHEY_SIMPLEX
    draw = cv2.putText(img, label, (boundingBoxes[i][0], boundingBoxes[i][1]), font, 0.5, (0, 0, 255), 1,
                       cv2.LINE_AA)

cv2.imshow('Solution: {}'.format(tp_idx), draw)

cv2.waitKey(0)

#################################################################################

# Cuvamo resenje u izlazni fajl
cv2.imwrite("tests/{}_out2.png".format(tp_idx), draw)
