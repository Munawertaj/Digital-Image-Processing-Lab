import cv2
import matplotlib.pyplot as plt
import numpy as np


def addSaltAndPepperNoise(image):

    noiseIntensity = 0.3  # % of the pixels will be affected
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
                    if i >= 0 and i < height and j >= 0 and j < width:
                        sum += image[i][j] / val

            avgfilterImage[row][col] = sum

    return avgfilterImage


def medianFiltering(image, mask):

    height, width = image.shape
    medImg = image.copy()

    for row in range(height):
        for col in range(width):
            r = row - (mask // 2)
            c = col - (mask // 2)

            tempArr = []
            for i in range(r, r + mask, 1):
                for j in range(c, c + mask, 1):
                    if i >= 0 and i < height and j >= 0 and j < width:
                        tempArr.append(image[i][j])

            total = len(tempArr)
            tempArr.sort()
            medImg[row][col] = tempArr[total // 2]

    return medImg


def plotImage(image, x, y, z, imageTitle):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(x, y, z)
    plt.imshow(img)
    plt.title(imageTitle)


def calculatePsnr(original_img, processed_img, max_pixel_value):
    mse = np.mean((original_img - processed_img) ** 2)
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    return psnr


def Main():

    image = cv2.imread("Task-03/pcb.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (512, 512))

    plotImage(image, 2, 2, 1, "Original Image")

    noisyImage = addSaltAndPepperNoise(image)
    plotImage(noisyImage, 2, 2, 2, "Noisy Image")

    mask = 3

    avgImg = averageFiltering(noisyImage, mask)
    plotImage(avgImg, 2, 2, 3, "Average Filter Image")

    medianImg = medianFiltering(image, mask)
    plotImage(medianImg, 2, 2, 4, "Median Filter Image")

    print("Noise image PSNR = " + str(calculatePsnr(image, noisyImage, 255)))
    print("Average filter PSNR = " + str(calculatePsnr(image, avgImg, 255)))
    print("Median filter PSNR = " + str(calculatePsnr(image, medianImg, 255)))

    plt.tight_layout()
    plt.show()


Main()
