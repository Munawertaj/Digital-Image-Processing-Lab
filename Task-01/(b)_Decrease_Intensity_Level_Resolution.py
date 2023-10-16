import cv2
import matplotlib.pyplot as plt


def plotImage(image, imageTitle, r, c, x):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(r, c, x)
    plt.imshow(img)
    plt.title(imageTitle)


def decreaseIntensityResolution(image, bits):

    height, width = image.shape
    lowLevelImage = image.copy()

    intensityLevels = 2 ** bits

    for i in range(1, bits):
        intensityLevels //= 2
        step = 255 // (intensityLevels - 1)

        for row in range(height):
            for col in range(width):
                lowLevelImage[row, col] = (round(image[row, col] / step)) * step

        title = f"{intensityLevels}-Gray Levels"
        plotImage(lowLevelImage, title, 2, 4, i+1)


def Main():

    image = cv2.imread("Task-01/bird.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))

    plotImage(image, "Original Image (256-Gray Levels)", 2, 4, 1)

    bits = 8
    decreaseIntensityResolution(image, bits)

    plt.tight_layout()
    plt.show()


Main()
