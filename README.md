# faceDetection

Python project for face detection

## Requirements

Python

Opencv-python (cv2)

Mediapipe (mp)

Numpy (np)

Face_recognition

dlib\*

\*watch out for their docs on installing with cuda enabled (pain in the ass alert!!)

## Instructions

Run in command line the main.py\*

`python main.py -h`

\*For VSCode, it's already configured the launch.json for F5 starting the project

### remake

First to be run, it reads all images in the assets/models folder and create their encoding for the face match algorithm. It also saves the thumbnails in the outputs folder.

It was meant to run in the gpu and for now it's not an argument to change. It's necessary to have cuda installed, or it will default to the cpu and the 'cnn' algorithm for recognition would run slow.

After this run, it is supposed to create in assets/models a models.pkl file, which is a dump of the generated encodings for each file in models.

For now, it is only looking for "jpg" images and it uses the file name (if in the assests/models folder) or the folder name (if it's in assets/models/subfolder).

`python main.py remake`

### detect

After you have your models processed, this instruction runs through the files in assets/samples folder, and do the face matching. It will recognize all faces and compare to the ones in the assets/models.

For now, for each file, it will display a window with the faces recognized displaying their names or "unkown" otherwise.

`python main.py detect --save --save-unknown`

\*If you use the save flag, it will add to the models the matches.

### webcam

After you have your models processed, this instruction initializes a first webcam detected and live stream it to face recognition. It displays a window with the faces recognized displaying their names or "unkown" otherwise.

`python main.py webcam`

### fix

If during a run you chose to save the matches something went amiss, just run the fix with the index and the correct name, for the models to be updated.

`python main.py fix -i 10 -n person`
`python main.py fix -i 10 20 -n person person2`
`python main.py fix -i 10 11 -n person`

\* this will search for the 10 item in the models list, and update it with the correct name, and also move the thumbnail to the correct folder.

### view

Check the original file the originated one of the index in models.

`python main.py view 10`

### print

Prints all saved models with their id, path and name.

`python main.py print`

### clear

Removes from the models all unknowns that doesn't have ate least X occurencies. Where X is the threshold (default=10).

`python main.py clear`

## Improvements

I tried applying multiprocessing for the models generation, but it just bottlenecked the gpu, for some files I tried expected more than 12Gb of memory available. So, for now it was disabled.

[x] improve the models, reducing their size for an optimal run

    - It happened to have lower accuracy as well, so I kept the original file for the face_recognition and only reduced for the viewing or thumbnail

[] make a choice of only one lib for face recognition (at this point there are 2 libs for that in play - opencv-python and face_recognition)

    - OpenCV was kept at bay for now for the processing the faces, as using only the face_recognition for the detection and matches are more intuitive and compatible

[x] allow argument choices for cpu/gpu

    - use the --model flag to state your choice [cnn/hog]

[x] outputs the generated matches to file

    - currently saving a list of matches with name and their thumbnails

[] make the unknowns presentation interactive, so you can name them on the go
