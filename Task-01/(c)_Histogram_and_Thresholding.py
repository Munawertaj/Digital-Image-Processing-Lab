import cv2
import matplotlib.pyplot as plt
import numpy as np


def plotImage(image, imageTitle, r, c, x):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(r, c, x)
    plt.imshow(img)
    plt.title(imageTitle)


def showHistogram(histogram, histoTittle, x, y, z):
    plt.subplot(x, y, z)
    plt.bar(range(256), histogram, color="black", width=1)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title(histoTittle)
    plt.xlim([-5, 260])


def performThresholding(image, threshold):
    height, width = image.shape

    for row in range(height):
        for col in range(width):
            if image[row][col] > threshold:
                image[row][col] = 255
            else:
                image[row][col] = 0


def makeHistogram(image):
    height, width = image.shape
    histogram = np.zeros(256)
    for row in range(width):
        for col in range(height):
            histogram[image[row][col]] += 1

    histogram /= 512 * 512  # For normalized histogram
    return histogram


def Main():

    image = cv2.imread("Task-01/bird2.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (512, 512))

    plotImage(image, "Original Image", 2, 2, 1)

    histogram = makeHistogram(image)
    showHistogram(histogram, "Histogram of original image", 2, 2, 2)

    threshold = 127
    performThresholding(image, threshold)
    plotImage(image, "Threshold Image", 2, 2, 3)

    histogram = makeHistogram(image)
    showHistogram(histogram, "Histogram of Threshold image", 2, 2, 4)

    plt.tight_layout()
    plt.show()


Main()
