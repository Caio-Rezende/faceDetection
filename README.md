# faceDetection
 Python project for face detection

## Requirements
Python
Opencv-python (cv2)
Mediapipe (mp)
Numpy (np)
Face_recognition
dlib*

*watch out for their docs on installing with cuda enabled (pain in the ass alert!!)

## Instructions
Run with one argument in command line the main.py*
```python main.py [file,webcam,remake]```

*For VSCode, it's already configured the launch.json for F5 starting the project

### remake
First to be run, it reads all images in the assets/models folder and create their encoding for the face match algorithm.
It was meant to run in the gpu and for now it's not an argument to change. It's necessary to have cuda installed, or it will default to the cpu and the 'cnn' algorithm for recognition would run slow. 
After this run, it is supposed to create in assets/models a models.pkl file, which is a dump of the generated encodings for each file in models.
For now, it is only looking for "jpg" images and it uses the file name (if in the assests/models folder) or the folder name (if it's in assets/models/subfolder).

### file
After you have your models processed, this instruction runs through the files in assets/samples folder, and do the face matching. It will recognize all faces and compare to the ones in the assets/models.
For now, for each file, it will display a window with the faces recognized displaying their names or "unkown" otherwise.

### webcam
After you have your models processed, this instruction initializes a first webcam detected and live stream it to face recognition. It displays a window with the faces recognized displaying their names or "unkown" otherwise.

## Improvements
I tried applying multiprocessing for the models generation, but it just bottlenecked the gpu, for some files I tried expected more than 12Gb of memory available. So, for now it was disabled.
[] improve the models, reducing their size for an optimal run
[] make a choice of only one lib for face recognition (at this point there are 2 libs for that in play - opencv-python and face_recognition)
[] allow argument choices for cpu/gpu
[] outputs the generated matches to file
