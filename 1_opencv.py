##Importing Images in OpenCV------------------------------------------------------------------

#Import the necessary libraries
import numpy as np
import cv2 
import matplotlib.pyplot as plt 
%matplotlib inline

#Read in the image using the imread function.
img_raw = cv2.imread('mandrill_colour.png')

#The type and shape of the array.
type(img_raw) 
#numpy.ndarray
img_raw.shape 
#(1300, 1950, 3)

plt.imshow(img_raw)

img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB) 
plt.imshow(img)

------------------------------------------------------
import cv2
img = cv2.imread('X.jpg')
while True:
    cv2.imshow('mandrill',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
#press escape to get out
---------------------------------------------------
#The images can be saved in the working directory as follows:
cv2.imwrite('X.png',img)
--------------------------------------------------------

#Basic Operations on Images
#we will learn how we can draw various shapes on an existing image to get a flavour of working with OpenCV.

#Drawing on Images

#Begin by importing necessary libraries.
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import cv2

#Create a black image which will act as a template.
image_blank = np.zeros(shape=(512,512,3),dtype=np.int16)
#Display the black image.
plt.imshow(image_blank)

#Function & Attributes

#The generalised function for drawing shapes on images is:
#cv2.shape(line, rectangle etc)(image,Pt1,Pt2,color,thickness)

# Draw a diagonal red line with thickness of 5 px
line_red = cv2.line(image_blank,(0,0),(511,511),(255,0,0),5)
plt.imshow(line_red)

# Draw a diagonal yellow line with thickness of 5 px
line_X = cv2.line(image_blank,(511,0),(0,511),(255,220,0),5)
plt.imshow(line_X)

# Draw a diagonal green line with thickness of 5 px
line_green = cv2.line(image_blank,(0,0),(511,511),(0,255,0),5)
plt.imshow(line_green)

#Draw a blue rectangle with a thickness of 5 px
rectangle= cv2.rectangle(image_blank,(384,0),(510,128),(0,0,255),5)
plt.imshow(rectangle)

circle = cv2.circle(image_blank,(447,63), 63, (255,220,0), -1) # -1 corresponds to a filled circle
plt.imshow(circle)

#writing on images
font = cv2.FONT_HERSHEY_SIMPLEX
text = cv2.putText(image_blank,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
plt.imshow(text)







