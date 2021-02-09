# Drowsiness Detector  ( Computer vision Fun project :) )

## Description
This project involves detecting the drowsiness of the user based on facial appearance. For the project, I have used shape_predictor_68_face_landmarks.dat. in order to detect the facial landmark points.

## shape_predictor_68_face_landmarks.dat.
This is trained on the ibug 300-W dataset (https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)

 ```text
C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
300 faces In-the-wild challenge: Database and results. 
Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
 ```
 
The license for this dataset excludes commercial use and Stefanos Zafeiriou, one of the creators of the dataset, asked me to include a note here saying that the trained model therefore can't be used in a commerical product. So you should contact a lawyer or talk to Imperial College London to find out if it's OK for you to use this model in a commercial product.

Also note that this model file is designed for use with dlib's HOG face detector. That is, it expects the bounding boxes from the face detector to be aligned a certain way, the way dlib's HOG face detector does it. It won't work as well when used with a face detector that produces differently aligned boxes, such as the CNN based mmod_human_face_detector.dat face detector.

## Method
1. Detect the facial landmark points on each images using shape_predictor_68_face_landmarks.dat.
2. It is used to locate the position of 68 coordinates ( x ,y)  mapping the facial points on the face of a person as image shown.
3. Get the landmarks(x,y coordinates) of upper and lower lip. Similarly for right and left eyes.
4. Calculate the mean (x, y) coordinates of upper and lower lip. Similarly, calculate mean for eye coordinates.
5. Calculate the distance between lips and distance between eye lids and compare with threshold value. ( I have used 15 for lips and 3 for eye lids).
6. If the distance is greater than the threshold, then yawn, eyes open and close is made.

## Installation

### On Ubuntu

1. Set your python 3 as default if not.

    ```text
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2
    sudo update-alternatives --config python
    ```

2. git clone https://gitlab.com/pprasathpp/DrowsinessDetector.git
3. pip install virtualenv
4. enter command : virtualenv your_choice_env
5. source yout_choice_env/bin/activate
6. Check that you have a clean virtual environment

    ```text
    pip freeze
    ```

    you should get a empty line  if not check that your PYTHONPATH empty is, if not enter in your terminal PYTHONPATH="" and put the same command at the end of your .bashrc

7. pip install -r Requirements.txt
8. DONE  :blush:

## Usage

python3 drowsiness_detector.py

## samples
![Eyes open](/images/eyesOpen.png)
![Yawning when eyes are opened](/images/YawningEyesOpen.png)
![Yawning when eyes are opened](/images/YawningEyesClosed.png)


## Current work

Yawn detection and number of times yawn detected(count) has been implemented.
eye open and close detection 
