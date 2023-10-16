import cv2
import matplotlib.pyplot as plt


def plotImage(image, imageTitle, r, c, x):

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(r, c, x)
    plt.imshow(img)
    plt.title(imageTitle)


def differnceImage(image1, image2):

    height, width = image1.shape
    diffImage = image1.copy()

    for row in range(height):
        for col in range(width):
            diffImage[row][col] = abs(int(image1[row][col]) - int(image2[row][col]))

    return diffImage


def bitSlicing(image, mask):

    height, width = image.shape
    updateImage = image.copy()

    for row in range(height):
        for col in range(width):
            updateImage[row][col] &= mask

    return updateImage


def Main():

    image = cv2.imread("Task-02\Flower.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (512, 512))

    plotImage(image, "Original Image", 1, 3, 1)

    mask = 224

    updateImage = bitSlicing(image, mask)
    plotImage(updateImage, "Update Image", 1, 3, 2)

    diffImage = differnceImage(image, updateImage)
    plotImage(diffImage, "Difference Image", 1, 3, 3)

    plt.tight_layout()
    plt.show()


Main()
