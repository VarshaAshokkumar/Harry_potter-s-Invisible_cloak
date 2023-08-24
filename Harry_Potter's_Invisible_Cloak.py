#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import time
import numpy as np


# In[2]:


cap = cv2.VideoCapture(0)


# In[ ]:


_, background = cap.read()
time.sleep(2)
_, background = cap.read()
open_kernel = np.ones((5,5),np.uint8)
close_kernel = np.ones((7,7),np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)
def filter_mask(mask):
    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    open_mask = cv2.morphologyEx(close_mask, cv2.MORPH_OPEN, open_kernel)
    dilation = cv2.dilate(open_mask, dilation_kernel, iterations= 1)
    return dilation
while cap.isOpened():
    ret, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([50, 80, 50])     
    upper_bound = np.array([90, 255, 255])  
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = filter_mask(mask)
    cloak = cv2.bitwise_and(background, background, mask=mask)
    
    inverse_mask = cv2.bitwise_not(mask)  
    current_background = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    combined = cv2.add(cloak, current_background)
    cv2.imshow("Final output", combined)
    if cv2.waitKey(1) == ord('q'):
        break
        cap.release()
        cv2.destroyAllWindows()


# In[ ]:




