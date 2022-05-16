import os
import cv2
import argparse
import numpy as np
import tkinter as tk
from tkinter import ttk
import tensorflow as tf
from keras import layers
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB7

"""My environment"""
# pip install tensorflow==2.3.0
# pip install keras==2.4.3


# To check if GPU is available or else you need to configure the environment!
print("Number of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

"""Arguments"""
parser = argparse.ArgumentParser(description='Driver Drowsiness Detection')
parser.add_argument("--Seed", default=42, type=int)
parser.add_argument("--Image_size", default=145, type=int)
parser.add_argument("--Batch_size_train", default=20, type=int)
parser.add_argument("--maximum_epochs", default=200, type=int)
parser.add_argument("--Test_size_ratio", default=0.15, type=float)
args = parser.parse_args()

"""Variables"""
arguments = parser.parse_args()
seed = arguments.Seed
IMG_SIZE = arguments.Image_size
batch_size_train = arguments.Batch_size_train  # You can change as per hardware limitations
max_epochs = arguments.maximum_epochs
test_size = arguments.Test_size_ratio
#########################

labels = os.listdir("../DriverDrowsiness/dataset/train")


# for yawn and not_yawn. Take only face
def face_for_yawn(direc="../DriverDrowsiness/dataset/train",
                  face_cas_path="../DriverDrowsiness/prediction_images/haarcascade_frontalface_default.xml"):
    yaw_no = []
    IMG_SIZE = 145
    categories = ["yawn", "no_yawn"]
    for category in categories:
        path_link = os.path.join(direc, category)
        class_num1 = categories.index(category)
        # print(class_num1)
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier(face_cas_path)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_color = img[y:y + h, x:x + w]
                resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
                yaw_no.append([resized_array, class_num1])
    return yaw_no


yawn_no_yawn = face_for_yawn()


# for closed and open eye
def get_data(dir_path="../DriverDrowsiness/dataset/train/",
             face_cas="../DriverDrowsiness/prediction-images/haarcascade_frontalface_default.xml",
             eye_cas="../input/prediction-images/haarcascade_eye.xml"):
    labels = ['Closed', 'Open']
    IMG_SIZE = 145
    data = []
    for label in labels:
        path = os.path.join(dir_path, label)
        class_num = labels.index(label)
        class_num += 2
        # print(class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(e)
    return data


data_train = get_data()


# extend data and convert array
def append_data():
    yaw_no = face_for_yawn()
    data = get_data()
    yaw_no.extend(data)
    return np.array(yaw_no)


# new variable to store
new_data = append_data()

# separate label and features
X = []
y = []
for feature, label in new_data:
    X.append(feature)
    y.append(label)

# reshape the array
X = np.array(X)
X = X.reshape((-1, IMG_SIZE, IMG_SIZE, 3))

# LabelBinarizer
"""
lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])
LabelBinarizer()
lb.classes_array([1, 2, 4, 6])
lb.transform([1, 6])
array([[1, 0, 0, 0],
       [0, 0, 0, 1]])
"""
label_bin = LabelBinarizer()
y = label_bin.fit_transform(y)
# label array
y = np.array(y)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)


def train_predict():
    """Train the model"""

    global mess

    # Dataloader and data Augmentation
    train_generator = ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.2, horizontal_flip=True, rotation_range=20,
                                         vertical_flip=False, fill_mode="nearest")
    test_generator = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_generator.flow(tf.convert_to_tensor(np.array(X_train)), tf.convert_to_tensor(y_train),
                                           shuffle=False, batch_size=batch_size_train)
    test_generator = test_generator.flow(tf.convert_to_tensor(np.array(X_test)), tf.convert_to_tensor(y_test),
                                         shuffle=False, batch_size=1)

    inputs = layers.Input(shape=X_train.shape[1:])

    outputs = EfficientNetB7(
        include_top=True,
        weights=None,
        drop_connect_rate=0.2,
        input_shape=X_train.shape[1:],
        pooling=None,
        classes=4,
        classifier_activation="softmax",
    )(inputs)

    model = tf.keras.Model(inputs, outputs)  # model creation

    model.compile(loss="categorical_crossentropy", metrics=["accuracy"],
                  optimizer=tf.keras.optimizers.Adam(
                      learning_rate=0.00005,
                      beta_1=0.9,
                      beta_2=0.999,
                      epsilon=1e-07,
                      amsgrad=True,
                      name='Adam'))
    model.summary()  # know how many trainable params are there

    # save best model
    checkpoint_filepath = './DriverDrowsiness'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # training now
    history = model.fit(train_generator, epochs=max_epochs, validation_data=test_generator, shuffle=True,
                        validation_steps=len(test_generator), callbacks=[model_checkpoint_callback],
                        verbose=1)

    # training output
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))

    # plot out path
    plot_out_path = "../DriverDrowsiness/"
    # save the plots
    plt.figure(1)
    plt.plot(epochs, accuracy, "b", label="Training Accuracy")
    plt.plot(epochs, val_accuracy, "r", label="Validation Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.minorticks_on()
    strFile_acc = ".../DriverDrowsiness/Accuracy_Plot.png"
    if os.path.isfile(strFile_acc):
        os.remove(strFile_acc)
        plt.savefig(str(Path(plot_out_path, 'Accuracy_Plot.png')), bbox_inches='tight', format='png', dpi=300)
    else:
        plt.savefig(str(Path(plot_out_path, 'Accuracy_Plot.png')), bbox_inches='tight', format='png', dpi=300)

    plt.figure(2)
    plt.plot(epochs, loss, "b", label="Training Loss")
    plt.plot(epochs, val_loss, "r", label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.minorticks_on()
    strFile_loss = ".../DriverDrowsiness/Loss_Plot.png"
    if os.path.isfile(strFile_loss):
        os.remove(strFile_loss)
        plt.savefig(str(Path(plot_out_path, 'Loss_Plot.png')), bbox_inches='tight', format='png', dpi=300)
    else:
        plt.savefig(str(Path(plot_out_path, 'Loss_Plot.png')), bbox_inches='tight', format='png', dpi=300)

    # save model
    model.save("drowsiness_newB7.h5", overwrite=True)
    model.save("drowsiness_newB7.model", overwrite=True)
    # After training please run the predict.py file


if __name__ == "__main__":
    train_predict()
