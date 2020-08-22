# Face-Recognition

### Face Recognition ### 
Face recognition is a method of identifying or verifying the identity of an individual using their face. Face recognition systems can be used to identify people in photos, video, or in real-time.

There are multiple methods in which facial recognition systems work, but in general, they work by comparing selected facial features from given image with faces within a database. 

Here are few applicatios of Face Recognition:
1.  Access and security
2. Criminal identification
3. Smart Payments
4. Advertising
5. Healthcare

<img src="http://drive.google.com/uc?export=view&id=1ZP3KYynv75Y319hUWbHMWAPOr4sTD8Au" alt="Face Recognition Example Pic">


## Face Detection vs Face Recognition ##

**Face Detection** is identifying does the given picture has any face or not whereas **Face Recognition** is finding out whose face is there in the picture this is done by comparing the faces in the picture with the faces in database.

As we have seen the difference between the face recognition ad face detection this project works on *Face Recognition*.

### Prerequisites ###

Inorder to get this project working on system. We need to install the following:
1. Python 3 - 64 Bit
2. Visual Studio - 2019 (Community Edition)
3. Modules:
* cmake
* dlib
* face recognition
* numpy
* openCV - Python

*Note :* Inorder to get this project working and install all the modules perfectly python version should be above 3 and of 64 Bit version and we have install **Desktop development with C++** package inn visual Studio.

### Understanding Code ###

Coming to the code, first we need to import the following :
~~~ import face_recognition as fr
import os
import cv2
import numpy as np
from time import sleep

~~~
I will give a brief overview of the above modules

## OpenCV ##

OpenCV is the huge open-source library for the computer vision, machine learning, and image processing and now it plays a major role in real-time operation which is very important in today’s systems. By using it, one can process images and videos to identify objects, faces, or even handwriting of a human. When it integrated with various libraries, such as Numpuy, python is capable of processing the OpenCV array structure for analysis. To Identify image pattern and its various features we use vector space and perform mathematical operations on these features.

## face_recognition ##

Recognize and manipulate faces from Python or from the command line with
the world’s simplest face recognition library.
Built using dlib’s state-of-the-art face recognition
built with deep learning. The model has an accuracy of 99.38% on the
Labeled Faces in the Wild benchmark.

## OS ##

The OS module in python provides functions for interacting with the operating system. OS, comes under Python’s standard utility modules. This module provides a portable way of using operating system dependent functionality. The *os* and *os.path* modules include many functions to interact with the file system.

## Numpy ##

NumPy is a python library used for working with arrays.

It also has functions for working in domain of linear algebra, fourier transform, and matrices.

NumPy was created in 2005 by Travis Oliphant. It is an open source project and you can use it freely.

NumPy stands for Numerical Python.

## Time ## 
Here in time module we just using sleep function.

time.sleep(secs)
Suspend execution of the calling thread for the given number of seconds. The argument may be a floating point number to indicate a more precise sleep time. The actual suspension time may be less than that requested because any caught signal will terminate the sleep() following execution of that signal’s catching routine. Also, the suspension time may be longer than requested by an arbitrary amount because of the scheduling of other activity in the system.


And next we have functions in our code. This entrie code has three functions. I will shortly give description about what that function does.

## 1. get_encoded_faces():##

This function reads all the images in the faces folder and encodes all the images and returns the dictionary with name and encoded image. This encoding will help to compare features with test images. All the images are encoded with this function.

Here is the code

~~~
def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded

~~~

## 2.unknown_image_encoded(img) ##

This function encodes the data of test image that we have given to get the output. This function also encodes the image data similar to the above one but not compare the images.

Here is the code

~~~
def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding
~~~

## 3.classify_face(im) ##

This is final function which does the major task i,e: Comparing the images encoded in the dictonary returned by first function with the encoded image given by second function. And then displays the output by giving the names on the test image.

Here is the code

~~~
def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    # Display the resulting image
    while True:

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names 


print(classify_face("test.jpg"))


~~~

### Running Project on Machine ###

So, to run this function on machine we need to execute the following command on Command line intrepreter.

I have assumed that the folder face_rec which have face_rec.py(code) and the other files  is on desktop.

~~~ 
cd desktop
~~~

~~~ 
cd  face_rec
~~~

this the command to install all the modules given in text file at a time. Takes few minutes if you are installing them for first time.

~~~ 
pip install -r req.txt 
~~~

finally to run the code.

~~~
python face_rec.py 
~~~

### OUTPUT ###

I have given the following faces as reference.

<img src="http://drive.google.com/uc?export=view&id=1ZP3KYynv75Y319hUWbHMWAPOr4sTD8Au" alt="Faces Pic">

This is the final Ouput to the test image I have given. 

<img src="http://drive.google.com/uc?export=view&id=1ZP3KYynv75Y319hUWbHMWAPOr4sTD8Au" alt="Output Pic">


Yes!! We are Done.

**NOTE :** 
1. I have taken help of various online resources to do this project and also to write this read me file.
2. If the images are not displaying here please refer the files. I have also uploaded them.


                                                       ****** THANK YOU ******


















