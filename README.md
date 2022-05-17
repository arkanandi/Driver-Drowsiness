# Driver Drowsiness
## Introduction

Implementation of driver drowsiness detection with EfficientNetB7 in TensorFlow Keras and OpenCV.

## Installation
- Install the requirements: pip install -r requirements.txt

## Training

- Arguments which you change as desired:
    - Image size
    - Training batch size
    - Maximum epochs
    - Test data ratio
- Haarcascade XMLs saved in folder prediction_images
        - For face: haarcascade_frontalface_default.xml
        - For eye: haarcascade_eye.xml

## Training
Run the train.py to train the model with the below command:
- python .\train.py --Image_size 145 --Batch_size_train 20 --maximum_epochs 200 --Test_size_ratio 0.15

## Prediction

- Run the predict.py to test the model saved as drowsiness_newB7.h5

You can change the image and the conditions to check for drowsiness as required. This is just a basic application with more updates coming in the future.

## Loss and Accuracy Plots

![Loss](https://github.com/arkanandi/Driver-Drowsiness/blob/ba17185ba762eb6acbc9b524591b18c0fc33f775/Loss_Plot.png "Loss Plot")
![Accuracy](https://github.com/arkanandi/Driver-Drowsiness/blob/ba17185ba762eb6acbc9b524591b18c0fc33f775/Accuracy_Plot.png "Accuracy Plot")

## Citation

@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International conference on machine learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}
