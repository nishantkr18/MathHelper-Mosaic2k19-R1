#Created By: nishantKr18
##################################
#     #  #            #   ##      
# #   #  #            #  #  #     
#  #  #  # ##  # ##   #   ##      
#   # #  ##    ##     #  #  #     
#    ##  # ##  #      #   ##      
################################## 

import numpy as np
import pandas as pd
import cv2
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import adam
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import TensorBoard

training_data = np.load('training_data.npy', allow_pickle=True)

total = len(training_data)
train = training_data[:-(int(total*0.05))]
test = training_data[-(int(total*0.05)):]

X_train = np.array([i[0] for i in train]).reshape(-1, 45, 45, 1)
y_train=to_categorical([int(i[1]) for i in train])
X_test = np.array([i[0] for i in test]).reshape(-1, 45, 45, 1)
y_test=to_categorical([int(i[1]) for i in test])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model = Sequential()
model.add(Convolution2D(64, 3, 3, input_shape = (45, 45, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32, 3, 3, activation="relu"))
model.add(Convolution2D(32, 3, 3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(30, activation="softmax"))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer = "adam", metrics = ["accuracy"])

hist = model.fit(X_train, y_train,
                 validation_data = (X_test, y_test), nb_epoch=6, batch_size=128, callbacks=[TensorBoard(log_dir = 'logs/{}'.format(time.ctime()))])

model.save("full_model_onMyData.mnist")
