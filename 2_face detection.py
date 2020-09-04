#Load the necessary Libraries
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
%matplotlib inline

#Loading the image to be tested 
test_image = cv2.imread('m.jpg')
#RGB color image
img = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB) 
plt.imshow(img)

#Converting to grayscale 
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
# Displaying the grayscale image 
plt.imshow(test_image_gray, cmap='gray')

#function to convert grey scale to RGB
def convertToRGB(image): 
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Loading the classifier for frontal face
haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.4, minNeighbors = 5);
# Let us print the no. of faces found
print('Faces found: ', len(faces_rects))
#Faces found:  1

#green rectangles around found faces
for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 15)
     
#convert image to RGB and show image
plt.imshow(convertToRGB(test_image))

#Face Detection with a generalized function-----------------------------------------------------------------------------------
#Load the necessary Libraries
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
%matplotlib inline

def detect_faces(cascade, test_image, scaleFactor = 1.1):
# create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()
    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)
    return image_copy

#testing
#loading image
test_image2 = cv2.imread('baby1.png')
# Converting to grayscale
test_image_gray_2 = cv2.cvtColor(test_image2, cv2.COLOR_BGR2GRAY)
# Displaying grayscale image
plt.imshow(test_image_gray_2, cmap='gray')

haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

#call the function to detect faces
faces = detect_faces(haar_cascade_face, test_image2)

#function to convert grey scale to RGB
def convertToRGB(image): 
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#convert to RGB and display image
plt.imshow(convertToRGB(faces))


# Face Detection on group image------------------------------------------------------------------------------------------------
#Load the necessary Libraries
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
%matplotlib inline

#Loading the image to be tested 
test_image = cv2.imread('group.png')
#RGB color image
img = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB) 
plt.imshow(img)

#Converting to grayscale 
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
# Displaying the grayscale image 
plt.imshow(test_image_gray, cmap='gray')

#function to convert grey scale to RGB
def convertToRGB(image): 
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Loading the classifier for frontal face
haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.001, minNeighbors = 2, maxSize = (30,30));------ -1 face
#faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.001, minNeighbors = 2, maxSize = (27,27));------ -2 faces
faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.001, minNeighbors = 16, maxSize = (25,25));
# Let us print the no. of faces found
print('Faces found: ', len(faces_rects))
#faces found: 7/15

#green rectangles around found faces
for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
     
#convert image to RGB and show image
plt.imshow(convertToRGB(test_image))
#more faces can still be detected with further tuning


