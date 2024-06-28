Object Detection
 Master
 By: Alexa Orrico, Software Engineer at Holberton School
 Weight: 5
 Migrated to checker v2: 
 Your score will be updated as you progress.
 Manual QA review must be done (request it when you are done with the project)


Resources
Read or watch:

OpenCV
Getting Started with Images
Object Localization
Landmark Detection
Object Detection
Convolutional Implementation Sliding Windows
Intersection Over Union
Nonmax Suppression
Non-Maximum Suppression for Object Detection in Python
Anchor Boxes
YOLO Algorithm
You Only Look Once: Unified, Real-Time Object Detection (CVPR 2016)
Real-time Object Detection with YOLO, YOLOv2 and now YOLOv3
Dive Really Deep into YOLO v3: A Beginner’s Guide
What’s new in YOLO v3?
What do we learn from single shot object detectors (SSD, YOLOv3), FPN & Focal loss (RetinaNet)?
mAP (mean Average Precision) for Object Detection
You Only Look Once (YOLO): Unified Real-Time Object Detection
Definitions to skim:

Object detection
References:

cv2.imshow
cv2.imread
cv2.line
cv2.putText
cv2.rectangle
cv2.resize
glob
You Only Look Once: Unified, Real-Time Object Detection
YOLO9000: Better, Faster, Stronger
YOLOv3: An Incremental Improvement
Advanced:

Image segmentation
Region Proposals
Understanding Feature Pyramid Networks for object detection (FPN)
What do we learn from region based object detectors (Faster R-CNN, R-FCN, FPN)?
Object Detection for Dummies Part 3: R-CNN Family
Image segmentation with Mask R-CNN
Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
What is OpenCV and how do you use it?
What is object detection?
What is the Sliding Windows algorithm?
What is a single-shot detector?
What is the YOLO algorithm?
What is IOU and how do you calculate it?
What is non-max suppression?
What are anchor boxes?
What is mAP and how do you calculate it?
Requirements
Python Scripts
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.9)
Your files will be executed with numpy (version 1.25.2) and tensorflow (version 2.15)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.11.1)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
All your files must be executable
The length of your files will be tested using wc
Download and Use OpenCV 4.9.0.80
alexa@ubuntu-xenial:~$ pip install --user opencv-python==4.9.0.80
alexa@ubuntu-xenial:~$ python3
>>> import cv2
>>> cv2.__version__
'4.9.0'
Test Files
yolo.h5
coco_classes.txt
yolo_images.zip
