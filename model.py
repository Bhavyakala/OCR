import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth= True
tf.keras.backend.set_session(tf.Session(config=config))
import os
import cv2
import json
from emnist import list_datasets, extract_training_samples

def model_arch(input_shape) :
    input_x = tf.keras.Input(input_shape)

    X = tf.keras.layers.Conv2D(64,(3,3),(2,2),data_format='channels_last')(input_x)
    X = tf.keras.layers.Conv2D(64,(3,3),(2,2))(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.MaxPooling2D((2,2),padding='same')(X)
    X = tf.keras.layers.Dropout(0.25)(X)
    
    X = tf.keras.layers.Conv2D(128,(3,3),(2,2))(X)
    X = tf.keras.layers.Conv2D(128,(3,3),(2,2))(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.MaxPooling2D((2,2),padding='same')(X)
    X = tf.keras.layers.Dropout(0.25)(X)

    X = tf.keras.layers.Conv2D(256,(3,3),(2,2),'same')(X)
    X = tf.keras.layers.Conv2D(256,(3,3),(2,2),'same')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.MaxPooling2D((2,2),padding='same')(X)
    X = tf.keras.layers.Dropout(0.5)(X)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(512,activation='relu')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(128,activation='relu')(X)
    X = tf.keras.layers.Dense(62,activation='softmax')(X)

    model = tf.keras.Model(inputs=input_x, outputs=X)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def train() :
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                                            rotation_range=40,
                                                            width_shift_range=0.2,
                                                            height_shift_range=0.2,
                                                            rescale=1./255,
                                                            shear_range=0.2,
                                                            zoom_range=0.5,
                                                            horizontal_flip=True,
                                                            vertical_flip=True,
                                                            fill_mode='nearest')
    train_generator = datagen.flow_from_directory('D:\Coding_wo_cp\OCR\English\Hnd\Img', 
                                                        target_size=(300,400),
                                                        batch_size=4,
                                                        class_mode='categorical')
    # valid_generator = datagen.flow_from_directory('D:/Coding_wo_cp/OCR/data/val', 
    #                                                     target_size=(128,128),
    #                                                     batch_size=32,
    #                                                     class_mode='categorical')
    # test_generator = datagen.flow_from_directory('D:/Coding_wo_cp/OCR/data/test', 
    #                                                     target_size=(128,128),
    #                                                     batch_size=16,
    #                                                     class_mode='categorical')                                                    
    return train_generator

train_generator = train()
model = model_arch((300,400,3))
model.summary()
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
# STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
# STEP_SIZE_TEST = test_generator.n//test_generator.batch_size


# with open('class_indices.json','w') as f:
#     json.dump(train_generator.class_indices,f)

DIR = 'model-'
os.mkdir(DIR)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./model-2/model_{epoch}.h5',
        save_best_only=True,
        monitor='loss',
        verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor = 'loss', 
                          min_delta = 0, 
                          patience = 5,
                          verbose = 1,
                          mode = 'auto'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor = 0.9, patience = 3)
]
model.fit_generator( generator=train_generator, 
                     steps_per_epoch=STEP_SIZE_TRAIN,
                     epochs=20,
                     callbacks=callbacks)

model.save('model-2/model-2-data_n.h5')
model = tf.keras.models.load_model('D:\Coding_wo_cp\OCR\model-1.h5')

# scores = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
# [0.3745442553009963, 0.87961155] data
# scores = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
# print(scores)

im = cv2.imread('D:/Coding_wo_cp/OCR/English/Hnd/Img/0/img001-001.png')