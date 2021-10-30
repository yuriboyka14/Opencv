import cv2
import numpy as np
import matplotlib.pyplot as plt

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


    def field_recognition(self):

        img = cv2.imread(self.image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 80, 150, apertureSize=3)

        self.set_dimensions(img)
        self.set_width()
        self.set_height()
        self.set_threshold()

        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=int(self.threshold))
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

        print(f"{lines}\n")

        cv2.imwrite('img/houghlines3.png', img)

        plt.imshow(img)
        plt.show()
        plt.imshow(edges)
        plt.show()
        cv2.imshow("edges image", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        vertical = []
        horizontal = []
        # first loop is responsible for establishing if the field fulfill basic criteria to become ttt field. number of
        # lines has to match, orientation has to match (theta is either 0 or 90 degrees - they have to be perpendicular
        # to each other). Then we divide them into vertical and horizontal (lines) groups.

        if lines_size == 8 or lines_size == 4:
            for i in range(0, lines_size):
                if lines[i][0][1] > 0.0 - 0.12 and lines[i][0][1] < 0.0 + 0.12:
                    vertical.append(lines[i][0])
                elif lines[i][0][1] > np.pi / 2 - 0.12 and lines[i][0][1] < np.pi / 2 + 0.12:   # around 7 degrees error
                    horizontal.append(lines[i][0])                                              # possible (0.12 radians
                else:                                                                           # is ~7 deg)
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


if __name__ == "__main__":
    checker = FieldChecker('img/sc2.png')
    checker.field_recognition()
