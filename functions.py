import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def shape_recognition(filename):
    image = cv2.imread(filename)

    gsc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # grayscale convertion

    edges = cv2.Canny(gsc, 30, 100)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)


    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (20, 220, 20), 3)

    plt.imshow(image)
    plt.show()



def shape_recognition_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, image = cap.read()
        print(type(image))
        # convert to grayscale
        grayscale = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # perform edge detection
        edges = cv2.Canny(grayscale, 30, 100)
        # detect lines in the image using hough lines technique
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 5, np.array([]), 50, 5)
        # iterate over the output lines and draw them


        # 4th argument in HoughLines (treshold) is important. It indicates the minimum number of pixels (?)
        # When there is less light in the room this value should be decreased



        print(type(lines))
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # show images
        cv2.imshow("image", image)
        cv2.imshow("edges", edges)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def circle_recognition(filename):
    img = cv2.imread(filename)

    # convert BGR to RGB to be suitable for showing using matplotlib library
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # make a copy of the original image
    cimg = img.copy()

    # convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply a blur using the median filter
    img = cv2.medianBlur(img, 5)

    # finds the circles in the grayscale image using the Hough transform
    circles = cv2.HoughCircles(image=img, method=cv2.HOUGH_GRADIENT, dp=0.9,
                               minDist=80, circles=np.array([]), param1=110, param2=39, maxRadius=70)

    for co, i in enumerate(circles[0, :], start=1):
        # draw the outer circle in green
        cv2.circle(cimg, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
        # draw the center of the circle in red
        cv2.circle(cimg, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)

    # print the number of circles detected
    print("Number of circles detected:", co)
    # save the image, convert to BGR to save with proper colors
    # cv2.imwrite("coins_circles_detected.png", cimg)
    # show the image
    plt.imshow(cimg)
    plt.show()


def circle_recognition_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()

        # convert BGR to RGB to be suitable for showing using matplotlib library
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # make a copy of the original image
        # cimg = image.copy()

        # convert image to grayscale
        cimg = img.copy()

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # apply a blur using the median filter
        img = cv2.medianBlur(img, 5)


        # finds the circles in the grayscale image using the Hough transform
        circles = cv2.HoughCircles(image=img, method=cv2.HOUGH_GRADIENT, dp=0.9,
                                   minDist=80, circles=np.array([[]]), param1=110, param2=39, maxRadius=70)
        # print(circles)
        # print(f"Type: {type(circles)}")
        # print(f"Shape: {circles.shape}")
        # # circles = np.arange(6).reshape((3, 2))
        # # print(f"Shape: {circles.shape}")
        # circles = np.uint8(np.around(circles))

        co = 0
        for i in enumerate(circles[0, :], start=1):
            # draw the outer circle in green
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), None)
            # draw the center of the circle in red
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), None)
            co += 1

        # print the number of circles detected
        print("Number of circles detected:", co)
        # save the image, convert to BGR to save with proper colors
        # cv2.imwrite("coins_circles_detected.png", cimg)
        # show the image
        plt.imshow(cimg)
        plt.show()

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def auxilary_function(image):

    img = cv2.imread(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((15, 1), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((17, 3), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(img, 175, 200)

    result = img.copy()
    lines = cv2.HoughLines(edges, 1, math.pi / 180.0, 165, np.array([]), 0, 0)
    a, b, c = lines.shape
    for i in range(a):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(result, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('aux/fabric_equalized_thresh.jpg', thresh)
    cv2.imwrite('aux/fabric_equalized_morph.jpg', morph)
    cv2.imwrite('aux/fabric_equalized_edges.jpg', edges)
    cv2.imwrite('aux/fabric_equalized_lines.jpg', result)

    cv2.imshow("thresh", thresh)
    cv2.imshow("morph", morph)
    cv2.imshow("edges", edges)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def auxilary_function_2(image):
    # kernel used for noise removal
    kernel = np.ones((7, 7), np.uint8)
    # Load a color image
    img = cv2.imread(image)
    # get the image width and height
    img_width = img.shape[0]
    img_height = img.shape[1]

    # turn into grayscale
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # turn into thresholded binary
    ret, thresh1 = cv2.threshold(img_g, 127, 255, cv2.THRESH_BINARY)
    # remove noise from binary
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

    # find and draw contours. RETR_EXTERNAL retrieves only the extreme outer contours
    contours, im2 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 15)

    cv2.imshow("result", thresh1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def auxilary_function_3(image):

    img = cv2.imread(image, 0)
    img = cv2.medianBlur(img, 5)
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    return th3


if __name__ == "__main__":
    shape_recognition_camera()