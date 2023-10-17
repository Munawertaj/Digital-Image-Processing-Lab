import cv2
import matplotlib.pyplot as plt
import numpy as np


def addSaltAndPepperNoise(image):
    noiseIntensity = 0.1  # % of the pixels will be affected
    noisyImage = image.copy()
    height, width = image.shape

    noisyPixels = int(noiseIntensity * height * width)
    noisyCoordinates = np.random.randint(0, high=height, size=(noisyPixels, 2))

    for pixel in noisyCoordinates:
        row, col = pixel
        if np.random.rand() < 0.5:
            noisyImage[row, col] = 0
        else:
            noisyImage[row, col] = 255

    return noisyImage


def averageFiltering(image, mask):
    height, width = image.shape
    avgfilterImage = image.copy()

    for row in range(height):
        for col in range(width):
            r = row - (mask // 2)
            c = col - (mask // 2)
            sum = 0
            val = mask * mask
            for i in range(r, r + mask, 1):
                for j in range(c, c + mask, 1):
                    x = (i + height) % height
                    y = (j + width) % width
                    sum += image[x][y] / val

            avgfilterImage[row][col] = sum

    return avgfilterImage


def calculatePsnr(original_img, processed_img, max_pixel_value):
    mse = np.mean((original_img - processed_img) ** 2)
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    return psnr


def plotImage(image, x, y, z, imageTitle):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(x, y, z)
    plt.imshow(img)
    plt.title(imageTitle)


def Main():

    image = cv2.imread("Task-03/pcb.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (512, 512))

    plotImage(image, 2, 3, 1, "Original Image")

    noisyImage = addSaltAndPepperNoise(image)
    plotImage(noisyImage, 2, 3, 2, "Noisy Image")

    mask = 3

    x = 4
    for i in range(mask, 8, 2):
        avgImg = averageFiltering(noisyImage, i)
        plotImage(avgImg, 2, 3, x, f"Average Filter Image of mask {i}*{i}")
        x += 1

    plt.tight_layout()
    plt.show()


Main()
