import cv2
import keras
import numpy as np
import sys

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
model = keras.models.load_model('model.h5')

# TODO
solution = img.copy()

#################################################################################

# Cuvamo resenje u izlazni fajl
cv2.imwrite("tests/{}_out.png".format(tp_idx), solution)