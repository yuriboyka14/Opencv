import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools


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

    # def findLines(self):
    #     cap = cv2.VideoCapture(0)
    #
    #     while True:
    #         ret, image = cap.read()
    #         print(type(image))
    #         # convert to grayscale
    #         gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    #         # perform edge detection
    #         edges = cv2.Canny(gray, 30, 100)
    #
    #         self.set_dimensions(image)
    #         self.set_width()
    #         self.set_height()
    #         self.set_threshold()
    #
    #
    #         # lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=int(self.threshold))
    #         # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 5, np.array([]), 50, 5)
    #         lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=170, lines=np.array([]))
    #
    #         # for i in range(0, len(lines)):
    #         #     a = np.cos(lines[i][0][1])  # theta
    #         #     b = np.sin(lines[i][0][1])  # theta
    #         #     x0 = a * lines[i][0][0]  # rho
    #         #     y0 = b * lines[i][0][0]  # rho
    #         #     x1 = int(x0 + 1000 * (-b))
    #         #     y1 = int(y0 + 1000 * a)
    #         #     x2 = int(x0 - 1000 * (-b))
    #         #     y2 = int(y0 - 1000 * a)
    #         if lines is not None:
    #             for line in lines:
    #                 a = np.cos(line[0][1])  # theta
    #                 b = np.sin(line[0][1])  # theta
    #                 x0 = a * line[0][0]  # rho
    #                 y0 = b * line[0][0]  # rho
    #                 x1 = int(x0 + 1000 * (-b))
    #                 y1 = int(y0 + 1000 * a)
    #                 x2 = int(x0 - 1000 * (-b))
    #                 y2 = int(y0 - 1000 * a)
    #
    #                 cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #
    #                 cv2.imshow("image", image)
    #                 cv2.imshow("edges", edges)
    #         else:
    #             cv2.imshow("image", image)
    #             cv2.imshow("edges", edges)
    #
    #         if cv2.waitKey(1) == ord("q"):
    #             break
    #
    #     cap.release()
    #     cv2.destroyAllWindows()
    #
    #     lines_size = len(lines)
    #
    #     self.showImages(lines, image, gray, edges)
    #
    #     return lines, lines_size

        # img = cv2.imread(self.image)
        # # img = cv2.medianBlur(img, 5)
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #
        # thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        # th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # th4 = cv2.threshold(th3, 127, 255, cv2.THRESH_BINARY)[1]


        # blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # edges = cv2.Canny(thresh, 80, 150, apertureSize=3)
        # edges = cv2.Canny(th3, 80, 150, apertureSize=3)
        # edges = cv2.Canny(blur, 80, 150, apertureSize=3)

        # images = [thresh, th3, edges, th4]
        #         # titles = ["regular threshold", "adaptive threshold", "canny edges", "adaptive + regular"]
        #         # for i in range(4):
        #         #     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        #         #     plt.title(f"{titles[i]}")
        #         #     plt.xticks([]), plt.yticks([])
        #         # plt.show()


        # --- changes up to this part





    def delete_redundant_lines(self, array):

        array_new = []
        max_difference = 0

        for line1, line2 in itertools.combinations(array, 2):
            if int(abs(line1[0] - line2[0])) > max_difference:
                max_difference = int(abs(line1[0] - line2[0]))
                line_1 = line1
                line_2 = line2
        array_new.append(line_1)
        array_new.append(line_2)

        return array_new


    def showImages(self, lines, img, gray, edges):
        print(f"{lines}\n")

        cv2.imshow("img", img)
        cv2.imshow("grey", gray)
        cv2.imshow("edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def field_recognition(self):

        cap = cv2.VideoCapture(1)

        vertical = []
        horizontal = []

        alignmError = 0.24  # ~14 degrees

        while True:
            ret, image = cap.read()
            print(type(image))
            # convert to grayscale
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            # perform edge detection
            edges = cv2.Canny(gray, 30, 100)

            self.set_dimensions(image)
            self.set_width()
            self.set_height()
            self.set_threshold()


            # lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=int(self.threshold))
            # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 5, np.array([]), 50, 5)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100, lines=np.array([]))

            # for i in range(0, len(lines)):
            #     a = np.cos(lines[i][0][1])  # theta
            #     b = np.sin(lines[i][0][1])  # theta
            #     x0 = a * lines[i][0][0]  # rho
            #     y0 = b * lines[i][0][0]  # rho
            #     x1 = int(x0 + 1000 * (-b))
            #     y1 = int(y0 + 1000 * a)
            #     x2 = int(x0 - 1000 * (-b))
            #     y2 = int(y0 - 1000 * a)
            if lines is not None:
                for line in lines:
                    a = np.cos(line[0][1])  # theta
                    b = np.sin(line[0][1])  # theta
                    x0 = a * line[0][0]  # rho
                    y0 = b * line[0][0]  # rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * a)

                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    cv2.imshow("image", image)
                    cv2.imshow("edges", edges)

                lines_size = len(lines)

                # first loop is responsible for establishing if the field fulfill basic criteria to become ttt field. number of
                # lines has to match, orientation has to match (theta is either 0 or 90 degrees - they have to be perpendicular
                # to each other). Then we divide them into vertical and horizontal (lines) groups.

                for i in range(0, lines_size):
                    if lines[i][0][1] > 0.0 - alignmError and lines[i][0][1] < 0.0 + alignmError or lines[i][0][
                        1] > np.pi - alignmError and lines[i][0][1] < np.pi + alignmError:
                        vertical.append(lines[i][0])
                    elif lines[i][0][1] > np.pi / 2 - alignmError and lines[i][0][1] < np.pi / 2 + alignmError:
                        horizontal.append(lines[i][0])
                    else:
                        print("\nIt is not the real tic-tac-toe field! Orientation of lines does not match.")
                        # return False

                horizontal = self.delete_redundant_lines(horizontal)
                vertical = self.delete_redundant_lines(vertical)

                if len(vertical) != len(horizontal):
                    print(f"\nIt is not the real tic-tac-toe field! Number or orientation (or both) does not match.")
                    # return False

                counter = 0
                for i in range(0, len(vertical)):
                    if (self.width / 3) - (7 * self.width / 100) < vertical[i][0] < (
                            self.width / 3) + (7 * self.width / 100):
                        counter += 1
                    elif (2 * self.width / 3) - (7 * self.width / 100) < vertical[i][0] < (
                            2 * self.width / 3) + (7 * self.width / 100):
                        counter += 1
                    else:
                        print("\nIt is not the real tic-tac-toe field! Lines are placed improperly.")
                        # return False

                for i in range(0, len(horizontal)):
                    if (self.height / 3) - (7 * self.height / 100) < horizontal[i][0] < (
                            self.width / 3) + (7 * self.width / 100):
                        counter += 1
                    elif (2 * self.height / 3) - (7 * self.height / 100) < horizontal[i][0] < (
                            2 * self.height / 3) + (7 * self.height / 100):
                        counter += 1
                    else:
                        print(counter)
                        print("\nIt is not the real tic-tac-toe field! Lines are placed improperly.")
                        # return False

                if counter == lines_size:
                    print("\nThis is a tic-tac-toe field! All requirements are fulfilled")
                    return True

            else:
                cv2.imshow("image", image)
                cv2.imshow("edges", edges)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    checker = FieldChecker('img/sc3.png')
    checker.field_recognition()
    # checker.auxilary_function()
    # checker.auxilary_function_2()
    # checker.auxilary_function_3()
