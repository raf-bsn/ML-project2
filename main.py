import cv2
import keras
import numpy as np
import sys
import random as rng
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
model = keras.models.load_model('fashion.h5')

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

ret,src_gray = cv2.threshold(src_gray, 240, 255, cv2.THRESH_BINARY)

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 5 :
                return True
            elif i==row1-1 and j==row2-1:
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
                if dist == True:
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

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
                     (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
        # cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

    cv2.imshow('Contours', drawing)
    return boundRect

source_window = 'Source'
cv2.namedWindow(source_window)
cv2.imshow(source_window, img)

max_thresh = 255
thresh = 10 # initial threshold
cv2.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
boundingBoxes = thresh_callback(thresh)

labelNames = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

kernel_sharpen = np.asarray([
  [-1, -1, -1],
  [-1, 9, -1],
  [-1, -1, -1]
])

kernel_gaussian_blur = np.asarray([
  [1, 4, 6, 4, 1],
  [4, 16, 24, 16, 4],
  [6, 24, 36, 24, 6],
  [4, 16, 24, 16, 4],
  [1, 4, 6, 4, 1]
])/256

kernel_gaussian_unmasking = np.asarray([
  [1, 4, 6, 4, 1],
  [4, 16, 24, 16, 4],
  [6, 24, -476, 24, 6],
  [4, 16, 24, 16, 4],
  [1, 4, 6, 4, 1]
])/(-256)

print(boundingBoxes)
# print(imageWidth, imageHeight)

i = 2

imageHeight = boundingBoxes[i][3]
imageWidth = boundingBoxes[i][2]

if imageWidth > imageHeight:
    # boundingBoxes[7][1] -= (imageWidth - imageHeight)
    boundingBoxes[i] = (boundingBoxes[i][0], boundingBoxes[i][1] - (imageWidth - imageHeight) // 2, boundingBoxes[i][2], boundingBoxes[i][3])
    firstImage = solution[boundingBoxes[i][1]: boundingBoxes[i][1] + boundingBoxes[i][2],
                 boundingBoxes[i][0]: boundingBoxes[i][0] + boundingBoxes[i][2]]
    res = cv2.resize(firstImage, None, fx=28 / imageWidth, fy=28 / imageWidth, interpolation=cv2.INTER_AREA)
else:
    # boundingBoxes[7][0] -= imageHeight - imageWidth
    boundingBoxes[i] = (boundingBoxes[i][0] - (imageHeight - imageWidth) // 2, boundingBoxes[i][1], boundingBoxes[i][2], boundingBoxes[i][3])
    firstImage = solution[boundingBoxes[i][1] : boundingBoxes[i][1] + boundingBoxes[i][3],
             boundingBoxes[i][0] : boundingBoxes[i][0] + boundingBoxes[i][3]]
    res = cv2.resize(firstImage, None, fx=28 / imageHeight, fy=28 / imageHeight, interpolation=cv2.INTER_AREA)

# firstImage = cv2.filter2D(firstImage, -1, kernel_sharpen)
# firstImage = cv2.filter2D(firstImage, -1, kernel_gaussian_unmasking)

res = cv2.bitwise_not(res)

cv2.imwrite('test.png', res)
plt.subplot(122),plt.imshow(res)
plt.show()

res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
res = res.reshape(1, 28, 28, 1)

if imageWidth > 13 and imageHeight > 13:
    probs = model.predict(res)
    print(probs)
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]
    print(label)

cv2.waitKey(0)

#################################################################################

# Cuvamo resenje u izlazni fajl
# cv2.imwrite("tests/{}_out.png".format(tp_idx), solution)