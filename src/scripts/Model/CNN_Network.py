from __future__ import print_function
import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.data import loadlocal_mnist
import cv2
from keras.models import load_model
from keras import regularizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from math import ceil
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from keras.layers.normalization import BatchNormalization
dirname = os.getcwd()
os.chdir(dirname)

import seaborn as sns


def loadlocal_mnist():
	loadlocal_mnist(
        images_path='train-images-idx3-ubyte', 
        labels_path='train-labels-idx1-ubyte')

def compile_and_train_model(batch_size, num_classes, numEpochs):
	
	# input image dimensions
	img_rows, img_cols = 28, 28

	# the data, split between train and test sets
	(x_train, y_train) = loadlocal_mnist(images_path='train-images-idx3-ubyte', labels_path='train-labels-idx1-ubyte')
	(x_test,y_test) = loadlocal_mnist(images_path='t10k-images-idx3-ubyte', labels_path='t10k-labels-idx1-ubyte')



	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	plt.title('Train Image')
	plt.imshow(x_train[6777].reshape(28,28))
	plt.show()

	model = Sequential()
	model.add(Conv2D(16,(5,5),padding="same",input_shape=input_shape,kernel_initializer="he_normal"))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))

	model.add(Conv2D(32,(3,3),padding="same",kernel_initializer="he_normal"))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64,(5,5),kernel_initializer="he_normal"))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))

	model.add(Conv2D(128,(3,3),padding="same",kernel_initializer="he_normal"))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256,(3,3),kernel_initializer="he_normal"))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))

	model.add(Conv2D(128,(2,2),kernel_initializer="he_normal"))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))

	model.add(Conv2D(64,(1,1),kernel_initializer="he_normal"))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))

	model.add(Conv2D(64,(1,1),kernel_initializer="he_normal",kernel_regularizer = regularizers.l2(0.01)))
	model.add(Flatten())
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
		          optimizer='adam',
		          metrics=['accuracy'])

	kf = KFold(n_splits=4) 
	datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.30, height_shift_range=0.30, zoom_range=0.20)
	for train_index, test_index in kf.split(x_train):
		
		print(train_index,test_index)
		x_train_split, x_test = x_train[train_index], x_train[test_index]
		y_train_split, y_test = y_train[train_index], y_train[test_index]
		model.fit_generator(datagen.flow(x_train_split, y_train_split,
		    shuffle=True, batch_size=batchSize), 
		    validation_data=(x_test, y_test),  
		    steps_per_epoch=len(x_train_split)/batchSize,
		    epochs=numEpochs, verbose=1) 

	score = model.evaluate(x_test, y_test, verbose=2)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])


	model.save('CNN_Model_Adv.h5')


if __name__=='__main__':

	loadlocal_mnist()
	compile_and_train_model(256,10,10)
	
