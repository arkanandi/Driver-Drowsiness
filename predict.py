from train import *

"""Predict on images"""

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.10)
test_generator = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_generator.flow(tf.convert_to_tensor(np.array(X_test)), tf.convert_to_tensor(y_test),
                                     shuffle=False, batch_size=1)

# prediction using the model --> You can change the image as they are present in the folder
model = tf.keras.models.load_model("../DriverDrowsiness/drowsiness_newB7.h5")
# Prediction
prediction = model.predict_generator(test_generator, steps=X_test.shape[0])

# classification report
labels_new = ["yawn", "no_yawn", "Closed", "Open"]

print(classification_report(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(prediction, axis=1),
                            target_names=labels_new, zero_division=0))


# predicting function
def prepare(filepath, face_cas="../DriverDrowsiness/prediction_images/haarcascade_frontalface_default.xml",
            eye_cas="../input/prediction-images/haarcascade_eye.xml"):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = img_array / 255
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


# yawn
prediction_yawn = model.predict([prepare("../DriverDrowsiness/dataset/train/yawn/9.jpg")])
print("Class:", labels_new[np.argmax(prediction_yawn)])
# no yawn
prediction_no_yawn = model.predict([prepare("../DriverDrowsiness/dataset/train/no_yawn/9.jpg")])
print("Class:", labels_new[np.argmax(prediction_no_yawn)])
# Eye Closed
prediction_eye_closed = model.predict([prepare("../DriverDrowsiness/dataset/train/Closed/_30.jpg")])
print("Class:", labels_new[np.argmax(prediction_eye_closed)])
# Eyes open
prediction_eye_open = model.predict([prepare("../DriverDrowsiness/dataset/train/Open/_30.jpg")])
print("Class:", labels_new[np.argmax(prediction_eye_open)])

NORM_FONT = ("Helvetica", 10)


def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    labelsss = ttk.Label(popup, text=msg, font=NORM_FONT)
    labelsss.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command=popup.destroy)
    B1.pack()
    popup.mainloop()


"""Check drowsiness --> Change the conditions as needed!"""
img_path = "../DriverDrowsiness/dataset/train/yawn/148.jpg"
predict = model.predict([prepare(img_path)])
pred_class = labels_new[np.argmax(predict)]
if pred_class == "Open" or pred_class == "no_yawn":
    mess = str("The driver is not drowsy")
    """Save the reports in a text file"""
    f = open("results.txt", "w")
    f.write(str(classification_report(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(prediction, axis=1),
                                      target_names=labels_new, zero_division=0)))
    f.write('\n')
    f.close()
    f = open("results.txt", "a")
    f.write(str(mess))
    f.close()
    # Click Okay in the pop up message box to terminate the exam
    popupmsg(mess)
elif pred_class == "Closed" or pred_class == "yawn":
    mess = str("The driver is drowsy")
    """Save the reports in a text file"""
    f = open("results.txt", "w")
    f.write(str(classification_report(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(prediction, axis=1),
                                      target_names=labels_new, zero_division=0)))
    f.write('\n')
    f.close()
    f = open("results.txt", "a")
    f.write(str(mess))
    f.close()
    # Click Okay in the pop up message box to terminate the exam
    popupmsg(mess)
else:
    # Click Okay in the pop up message box to terminate the exam
    popupmsg("Wrong class!")
