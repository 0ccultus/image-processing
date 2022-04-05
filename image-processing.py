import numpy as np
import cv2 as cv
from tkinter import *
from tkinter import ttk

"""
Small test program for messing with video feed processing.
Currently sets a color mask and can track a circle of the respective color, barely.

Author: Julian Ayres
Start Date: 2022.04.03
Python Version 3.10.4

Goals:
-Set up TK window that loads/sets configuration. checkboxes and the like
-face/hand recognition for funsies
-active color mask sliders

"""

def create_color_mask(colors, hsv_color_window, color_to_detect):
    hsv_lower = np.array(colors[0])
    hsv_upper = np.array(colors[1])
    if color_to_detect == "new_red":
        hsv_lower1 = np.array(colors[2])
        hsv_upper1 = np.array(colors[3])
        return cv.inRange(hsv_color_window, hsv_lower, hsv_upper) + cv.inRange(hsv_color_window, hsv_lower1, hsv_upper1)
    return cv.inRange(hsv_color_window, hsv_lower, hsv_upper)

def detect_circle(img, original):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #gray_blurred = cv.blur(gray, (3,3))
    detected_circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 2,30,
                                      param1 = 50, param2 = 25, minRadius = 10,
                                      maxRadius = 0)
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0,:1]:
            a,b,r = pt[0],pt[1],pt[2]
            cv.circle(original, (a,b),r,(0,255,0),2)
            cv.circle(original,(a,b),1,(0,0,255),3)

#Does nothing important thus far
def create_circle(img, angle):
    W = 400
    thickness = 2
    line_type = 8

    cv.ellipse(img,(W // 8, W // 8),(W // 8, W // 8),angle,0,360,(255, 0, 0),thickness,line_type)

def denoise(img):
    sat = cv.cvtColor(img, cv.COLOR_BGR2HSV)[:,:,1]
    thresh = cv.threshold(sat, 50, 255, cv.THRESH_BINARY)[1]
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9,9))
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)
    mask = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel, iterations=1)
    g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    otsu = cv.threshold(g, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    otsu_result = otsu.copy()
    otsu_result[mask==0] = 0
    img_result = img.copy()
    img_result[mask==0] = 0
    return img_result

#Main function. Hit Q to quit while the cameras are running.
def main():
    #choose between the colors listed in color_ranges. Make sure that the variable color_to_detect is a string.
    #These color values should probably be tuned better
    color_to_detect = "blue"

    #Tkinter hello world
    """
    root = Tk()
    frm = ttk.Frame(root, padding=10)
    frm.grid()
    ttk.Label(frm, text="Hello World!").grid(column=0, row=0)
    ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1,row=0)
    root.mainloop()
"""
    
    vid = cv.VideoCapture(0)

    color_ranges = {'red' : [[159,155,85],[179,255,255]],
                    'new_red' : [[0,100,50],[10,255,255],[160,100,50],[179,255,255]],
                    'blue' : [[110,50,50],[130,255,255]],
                    'green' : [[40,60,50],[80,255,255]],
                    'yellow' : [[25,35,50],[37,255,255]],
                    'orange' : [[15,50,50],[25,255,255]],
                    'purple' : [[125,50,50],[135,255,255]],
                    'pink' : [[160,100,100],[179,255,255]],
                }

    while(1):
        _,frame = vid.read()

        frame = cv.GaussianBlur(frame, (3,3),0)
        
        # Create various colorspace windows
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask = create_color_mask(color_ranges[color_to_detect], hsv, color_to_detect)
        mask2 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        
        # Put to rest for now
        """
        kernel = np.ones((5,5), np.uint8)
        img_dilated = cv.dilate(mask,kernel,iterations=0)
        img_eroded = cv.erode(mask,kernel,iterations=2)
        # mask with dilation + erosion
        mask3 = img_eroded
        cv.imshow('erosion',img_eroded)
        cv.imshow('dilation',img_dilated)
        """
        res = cv.bitwise_and(frame,frame,mask=mask)

        detect_circle(res, frame)
        
        stacked = np.hstack((mask2,frame,res))
        cv.imshow('Frame & Res',cv.resize(stacked,None,fx=0.8,fy=0.8))

        k = cv.waitKey(33) & 0xFF
        if k == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()
    print("Program exited.")

if __name__ == "__main__":
    main()
