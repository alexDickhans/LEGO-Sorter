#!/usr/bin/python3

import tensorflow.keras
import io
import time
import cv2
import numpy as np
from PIL import Image, ImageOps

def loadModel(filePath):
    """Load the Keras .h5 model
    """
    model = tensorflow.keras.models.load_model(filePath, compile=False)

    return model

def getWebcam(webcamPath):
    """Get a frame from the webcam

    Args:
        webcamPath (String): The path to the webcam

    Returns:
        PIL Image: the frame from the webcam
    """

    cap = cv2.VideoCapture(0) # says we capture an image from a webcam
    _,cv2_im = cap.read()
    
    try:
        cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    except Exception as e:
        print("Failed to get webcam. Error: ")
        print(e)      
    pil_im = Image.fromarray(cv2_im)
    pil_im.show()

    return pil_im

def prepareImage(image, width):
    normalImage = ImageOps.fit(image, (width, width), Image.ANTIALIAS)

    imageArray = np.asarray(normalImage)

    normalizedImageArray = (imageArray.astype(np.float32) / 127.0) - 1

    npData = np.ndarray(shape=(1, width, width, 3), dtype=np.float32)

    npData[0] = normalizedImageArray

    return npData