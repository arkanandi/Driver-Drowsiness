Driver Drowsiness
Environment:
    • tensorflow==2.3.0
    • keras==2.4.3
    • os
    • cv2
    • argparse
    • numpy
    • tkinter
    • pathlib
    • matplotlib
    • sklearn
Training:
    Arguments which you change as desired:
        • Image size
        • Training batch size
        • Maximum epochs
        • Test data ratio
Haarcascade XMLs saved in folder prediction_images:
    • For face: haarcascade_frontalface_default.xml
    • For eye: haarcascade_eye.xml

def face_for_yawn → to detect the face as ROI for yawn and no_yawn and create the classes as 0 and 1 respectively
def get_data → to create the classes for open and closed as 2 and 3 respectively
def append_data → To append data of all the classes into a single dataset

Separate label and features
Label Binarizer → To binarize the labels
Train test split → To split the data into train and test/validation data

def train_predict → To train the model
    • Model → EfficientNetB7
    • Activation function → Softmax
    • Loss → Categorical cross-entropy
    • Optimizer → Adam

Run the train.py to train the model and save the best checkpoints.

Prediction:
Run the predict.py to test the model saved as drowsiness_newB7.h5

You can change the image and the conditions to check for drowsiness as required.