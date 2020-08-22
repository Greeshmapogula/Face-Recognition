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












