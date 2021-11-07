import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# dodany komatarz dla sprawdzenia

class FieldChecker:
    def __init__(self, image):
        self.dimensions = None
        self.width = None
        self.height = None
        self.threshold = None
        self.image = image

    def set_dimensions(self, image):
        self.dimensions = image.shape

    def set_width(self):
        self.width = self.dimensions[0]

    def set_height(self):
        self.height = self.dimensions[1]

    def get_dimensions(self):
        return self.dimensions

    def set_threshold(self):
        threshold = (5 / 10) * self.width  # line has to have a length of at least 60% of longer dimension
        if threshold < (5 / 10) * self.height:
            threshold = (5 / 10) * self.height

        self.threshold = threshold

    def get_threshold(self):
        return self.threshold

    def findLines(self):
        img = cv2.imread(self.image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        thresh = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)[1]

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 80, 150, apertureSize=3)

        self.set_dimensions(img)
        self.set_width()
        self.set_height()
        self.set_threshold()

        # lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=int(self.threshold))
        # lines = cv2.HoughLines(edges, 1, math.pi / 180.0, 165, np.array([]), 0, 0)
        lines_size = len(lines)

        for i in range(0, lines_size):
            a = np.cos(lines[i][0][1])  # theta
            b = np.sin(lines[i][0][1])  # theta
            x0 = a * lines[i][0][0]     # rho
            y0 = b * lines[i][0][0]     # rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # PRINT THE RESULT
        self.showImages(lines, img, gray, edges)

        return lines, lines_size

    def showImages(self, lines, img, gray, edges):
        print(f"{lines}\n")

        cv2.imshow("img", img)
        cv2.imshow("grey", gray)
        cv2.imshow("edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def field_recognition(self):
        lines, lines_size = self.findLines()

        vertical = []
        horizontal = []
        # first loop is responsible for establishing if the field fulfill basic criteria to become ttt field. number of
        # lines has to match, orientation has to match (theta is either 0 or 90 degrees - they have to be perpendicular
        # to each other). Then we divide them into vertical and horizontal (lines) groups.

        alignmError = 0.24 # ~14 degrees

        if lines_size == 8 or lines_size == 4:
            for i in range(0, lines_size):
                if lines[i][0][1] > 0.0 - alignmError and lines[i][0][1] < 0.0 + alignmError:
                    vertical.append(lines[i][0])
                elif lines[i][0][1] > np.pi / 2 - alignmError and lines[i][0][1] < np.pi / 2 + alignmError:
                    horizontal.append(lines[i][0])
                else:
                    print("\nIt is not the real tic-tac-toe field! Orientation of lines does not match.")
                    return False
        else:
            print(f"\nIt is not the real tic-tac-toe field! Number of lines does not match. ({lines_size})")
            return False

        if len(vertical) != len(horizontal):
            print(f"\nIt is not the real tic-tac-toe field! Number or orientation (or both) does not match.")
            return False


        counter = 0
        for i in range(0, len(vertical)):
            if vertical[i][0] > (self.width / 3) - (7 * self.width / 100) and vertical[i][0] < (self.width / 3) + (7 * self.width / 100):
                counter += 1
            elif vertical[i][0] > (2 * self.width / 3) - (7 * self.width / 100) and vertical[i][0] < (2 * self.width / 3) + ( 7 * self.width / 100):
                counter += 1
            else:
                print("\nIt is not the real tic-tac-toe field! Lines are placed improperly.")
                return False

        for i in range(0, len(horizontal)):
            if horizontal[i][0] > (self.height / 3) - (7 * self.height / 100) and horizontal[i][0] < (self.width / 3) + (7 * self.width / 100):
                counter += 1
            elif horizontal[i][0] > (2 * self.height / 3) - (7 * self.height / 100) and horizontal[i][0] < (2 * self.height / 3) + (7 * self.height / 100):
                counter += 1
            else:
                print(counter)
                print("\nIt is not the real tic-tac-toe field! Lines are placed improperly.")
                return False

        if counter == lines_size:
            print("\nThis is a tic-tac-toe field! All requirements are fulfilled")
            return True


    def auxilary_function(self):
        # read image
        img = cv2.imread(self.image)

        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # threshold
        thresh = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)[1]

        # apply close to connect the white areas
        kernel = np.ones((15, 1), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((17, 3), np.uint8)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

        # apply canny edge detection
        edges = cv2.Canny(img, 175, 200)

        # get hough lines
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

        # save resulting images
        cv2.imwrite('aux/fabric_equalized_thresh.jpg', thresh)
        cv2.imwrite('aux/fabric_equalized_morph.jpg', morph)
        cv2.imwrite('aux/fabric_equalized_edges.jpg', edges)
        cv2.imwrite('aux/fabric_equalized_lines.jpg', result)

        # show thresh and result
        cv2.imshow("thresh", thresh)
        cv2.imshow("morph", morph)
        cv2.imshow("edges", edges)
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    checker = FieldChecker('img/sc3.png')
    checker.field_recognition()
    # checker.auxilary_function()
