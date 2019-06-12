import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Ucitavanje FashionMNIST skupa podataka
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Koristimo samo deo trening skupa (prvi od 10 fold-ova) radi efikasnosti treninga
skf = StratifiedKFold(n_splits=6, random_state=0, shuffle=False)
for train_index, test_index in skf.split(x_train, y_train):
    x_train, y_train = x_train[test_index], y_train[test_index]
    break
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#################################################################################
# U ovoj sekciji implementirati Keras neuralnu mrezu koja postize tacnost barem
# 85% na test skupu. Ne menjati fajl van ove sekcije.

x_train = x_train.reshape(10000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

batch_size = 32
num_classes = 10
epochs = 25

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling 2x2
model.add(Dropout(rate=0.25))  # Dropout metoda regularizacije

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(512))  # Prvi potpuno povezan sloj
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(num_classes))  # Finalni potpuno povezan sloj
model.add(Activation('softmax'))

# Koristimo SGD optimizer
# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = SGD(lr=0.01, momentum=0.9, decay=0.01 / epochs)

# Kompilacija modela
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Normalizacija podataka
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

# Ispisujemo finalni rezultat na test skupu
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#################################################################################

# Cuvanje istreniranog modela u fajl
model.save('fashion.h5')