#!/usr/bin/python3

import utils
import numpy
import colors

imageWidth = 224

if __name__ == "__main__":

    # Load the model
    model = utils.loadModel("model/keras_model.h5")

    # Get an image from the webcam
    image = utils.getWebcam("/dev/video0")
    normalImage = utils.prepareImage(image, imageWidth)

    # Run the prediction
    prediction = model.predict(normalImage)

    # get the largest value
    largest = max(prediction[0])
    index = int(numpy.where(prediction[0] == largest)[0])

    # Get the labels
    labels = utils.openLabels("model/labels.txt")
    label = utils.getLabel(labels, index)

    # Run the user defined function
    colors.runColor(label, index, prediction)
