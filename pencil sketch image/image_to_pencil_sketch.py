'''LGMVIP DATA SCIENCE INTERNSHIP'''
'''BASIC LEVEL TASK 4: IMAGE TO PENCIL SKETCH'''
'''PARTICIPANT NAME: DIENGA JOHN DIVINE '''
'''TECHNIQUE: COMPUTER VISION'''
import cv2 as cv

'''Reading the image'''
image = cv.imread('div.jpg')
cv.imshow('Div',image)

'''Resizing image'''
def resize(frame,scale=0.3):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimension = (width,height)
    return cv.resize(frame,dimension,interpolation=cv.INTER_AREA)
image_resized = resize(image)
#cv.imshow('Divine',image_resized)

'''Processing image'''
image_grayed = cv.cvtColor(image_resized,cv.COLOR_BGR2GRAY)
#cv.imshow('grayed image',image_grayed)

image_inverted = 255 - image_grayed
#cv.imshow('inverted image',image_inverted)

image_blured = cv.GaussianBlur(image_inverted,(21,21),0)
#cv.imshow('blur',image_blured)
blured_invert = 255 - image_blured
pencil_scketch = cv.divide(image_grayed,image_blured,scale=256.0)

'''Saving and displaying pencil image'''
cv.imwrite('sketch.jpg',pencil_scketch)
cv.imshow('sketch',pencil_scketch)

cv.waitKey(0)