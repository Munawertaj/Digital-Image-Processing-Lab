import cv2
import matplotlib.pyplot as plt
import numpy as np


def plotImage(image, imageTitle, r, c, x):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(r, c, x)
    plt.imshow(img)
    plt.title(imageTitle)


def decreaseSpatialResolution(image):

    height, width = image.shape

    for i in range(1, 9):
        height = height // 2
        width = width // 2
        downsampledImage = np.zeros((height, width), dtype=np.uint8)

        for row in range(height):
            for col in range(width):
                downsampledImage[row][col] = image[2 * row][2 * col]

        image = downsampledImage
        title = f"{width}*{height}"
        plotImage(image, title, 3, 3, i + 1)


def Main():

    image = cv2.imread("Task-01/cat.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))

    plotImage(image, "512*512", 3, 3, 1)

    decreaseSpatialResolution(image)

    plt.tight_layout()
    plt.show()


Main()
