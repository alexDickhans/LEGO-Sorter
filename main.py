#!/usr/bin/python3

import utils

imageWidth = 224

if __name__ == "__main__":
    model = utils.loadModel("model/keras_model.h5")
    image = utils.getWebcam("/dev/video0")
    normalImage = utils.prepareImage(image, imageWidth)

    # Run the prediction
    prediction = model.predict(normalImage)
    print(prediction[0])

