import numpy as np
import cv2

"""
The following code is supposed to display a temporal contrast view through
the devices front camera.
"""

# Make function for performing thresholding
def thresholding(pixel, thresh):
    if pixel > thresh: # ON event
        newPixel = 255 # Set pixel to white
    elif pixel < -thresh: # OFF event
        newPixel = 0 # Set pixel to black
    else: # NO event (-thresh < pixel < thresh)
        newPixel = 128 # Set pixel to gray
    return newPixel

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Feeds stream of numpy-array images to variable 'vid'
if not vid.isOpened():
    print('Cannot open camera.')
    exit()

# Each iteration of the while loop uses .read() to determine
# whether the next frame in the video stream being captured can
# be read (stored as 'ret') and stores the numpy array containing
# the individual frame for that iteration of the loop in variable
# 'frame'.
while(True):
    # Capture the video frame-by-frame.
    # The following returns a boolean indicating whether the next
    # frame could be read from the video object, and the respective
    # frame in the form of a numpy-array image.
    ret, frame1 = vid.read()
    ret, frame2 = vid.read()
    
    # Convert the color of each frame from RGB to grayscale. BGR == RGB
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    deltaFrame = gray2.astype(int) - gray1.astype(int)   # NOTE: IF YOU DONT MAKE YOUR DIFFERENCE AN INT, YOULL GET A LOT OF NOISE!!!
    # filteredFrame = np.zeros(gray1.shape)
    filteredFrame = gray1.copy() # Also, HAVE TO USE .copy() RATHER THAN np.zeros TO MAKE FILTERED FRAME OTHERWISE GRAY BECOMES WHITE FOR NO EVENT???
    threshold = 20

    # Cycle through each pixel and perform thresholding test
    # """
    for row in range(deltaFrame.shape[0]):
        for col in range(deltaFrame.shape[1]):
            filteredFrame[row][col] = thresholding(int(deltaFrame[row][col]), threshold)
    # """

    # Display the altered frame
    cv2.namedWindow('Filtered Livestream', cv2.WINDOW_NORMAL) # Make window and constrain its shape
    # cv2.namedWindow('Raw Livestream', cv2.WINDOW_NORMAL)

    cv2.imshow('Filtered Livestream', filteredFrame)
    # cv2.imshow('Raw Livestream', deltaFrame)
    
    # Break the infinite while loop by clicking in the display window
    # for the video feed and pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break
    
# When everything is done, release the capture
vid.release()
cv2.destroyAllWindows()
