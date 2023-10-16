import cv2
import matplotlib.pyplot as plt
import numpy as np


def plotImage(image, imageTitle, r, c, x):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(r, c, x)
    plt.imshow(img)
    plt.title(imageTitle)


def brightnessEnhancement(image, low, high, val):
    height, width = image.shape

    for row in range(height):
        for col in range(width):
            if image[row][col] >= low and image[row][col] <= high:
                image[row][col] += val


def Main():

    image = cv2.imread("Task-02\photographer.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (512, 512))

    plotImage(image, "Original Image", 1, 2, 1)

    low = 100
    high = 200
    val = 50

    brightnessEnhancement(image, low, high, val)
    plotImage(image, "Enhanced Image", 1, 2, 2)

    plt.tight_layout()
    plt.show()


Main()
