import cv2
import numpy as np
from game import findBestMove


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, gray = cv2.threshold(gray, 80, 230, cv2.THRESH_BINARY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 170)
    dillated_edges = cv2.dilate(edges.copy(), None, iterations=5)
    dillated_edges = cv2.morphologyEx(dillated_edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    return dillated_edges


def find_circle(processed_img, output_img, last_circles):
    rows = processed_img.shape[0]
    circles = cv2.HoughCircles(processed_img, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=10,
                               maxRadius=100)

    if circles is not None:
        circles = np.uint16(np.around(circles))

    return circles


def print_cricles(img, circles):
    if circles is not None:
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(img, center, radius, (187, 206, 125), 3)


def circle_and_x_position(circles, cnt, img, best_move_done):
    grid_return = np.zeros((3, 3), dtype=str)  # change to str from bool
    height = img.shape[1]
    xd = np.zeros((3, 3), dtype=bool)

    if cnt is not None:
        if circles is not None:
            leftup_x = cnt[3][0][0]
            leftup_y = cnt[3][0][1]

            rightup_x = cnt[0][0][0]
            rightup_y = cnt[0][0][1]

            leftdown_x = cnt[2][0][0]
            leftdown_y = cnt[2][0][1]

            rightdown_x = cnt[1][0][0]
            rightdown_y = cnt[1][0][1]

            for cr in circles[0, :]:
                (x, y) = (cr[0], cr[1])

                grid_x = np.zeros((3, 3), dtype=str)   # change to str from bool
                grid_y = np.zeros((3, 3), dtype=str)   # change to str from bool

                # grid_x
                if x < leftup_x:
                    grid_x[:][0] = "o"

                elif leftup_x <= x < rightdown_x:
                    grid_x[:][1] = "o"

                else:
                    grid_x[:][2] = "o"

                # grid_y
                if y > leftdown_y:
                    grid_y[0][:] = "o"

                elif leftdown_y >= y > rightup_y:
                    grid_y[1][:] = "o"

                else:
                    grid_y[2][:] = "o"

                grid_x = np.rot90(grid_x)
                grid_x = np.rot90(grid_x)
                grid_x = np.rot90(grid_x)

                # grid
                for t in range(3):
                    for r in range(3):
                        grid_return[t][r] = grid_return[t][r] or (grid_x[t][r] and grid_y[t][r])

                        # Position is not correct here - to be improved!
                        # Grid coordinates don't match with the actual (videocap) coordinates

        else:
            print("No circles!")
    else:
        print("No contour!")

    # --- added ---

    # grid with x
    if best_move_done:
        for i in range(len(best_move_done)):  # this loop checks if the field is empty and x can be applied
            if grid_return[best_move_done[i][0]][best_move_done[i][1]] == '':
                grid_return[best_move_done[i][0]][best_move_done[i][1]] = 'x'

    return grid_return


if __name__ == "__main__":

    vid = cv2.VideoCapture(0)
    last_contour = None
    last_circles = None

    best_move_done = []     # array which will contain positions of computer player

    x_pos = [[(100, 150), (600, 150), (1100, 150)],        # possible x positions (each square)
               [(100, 400), (600, 400), (1100, 400)],
               [(100, 650), (600, 650), (1100, 650)]]

    while True:
        ret, img = vid.read()
        processed_img = preprocess(img)

        contours, _ = cv2.findContours(processed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            accuracy = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, accuracy, True)

            if len(approx) == 4 and cv2.contourArea(approx) > 15000:
                last_contour = approx
                cv2.drawContours(img, [approx], -1, (120, 255, 155), 10)

            elif last_contour is not None:
                cv2.drawContours(img, [last_contour], -1, (150, 255, 155), 3)

        print_cricles(img, last_circles)

        k = cv2.waitKey(2)

        # --- added part ---

        last_circles = find_circle(processed_img, img, last_circles)

        # --- added part ---


        if k & 0xFF == ord('\r'):
            grid = circle_and_x_position(last_circles, last_contour, img, best_move_done)

            if grid is not None:
                print(f"Grid:\n {grid}")

            best_move = findBestMove(grid)
            best_move_done.append(best_move)

        if best_move_done:
            for i in range(len(best_move_done)):
                # print(f"general: {best_move_done}")
                # print(f"row: {best_move_done[i][0]}")
                # print(f"column: {best_move_done[i][1]}")
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'X', (x_pos[best_move_done[i][0]][best_move_done[i][1]]), font, 4, (200, 180, 100), 6, cv2.LINE_AA)
                                        # here we pass the coordinates for displaying x
                                        # in one of the available squares (depends on the best move)

        if k & 0xFF == ord('q'):
            if grid is not None:
                print(f"Grid:\n {grid}")

            break


        cv2.imshow('image', img)
        cv2.imshow('edges', processed_img)

    vid.release()
    cv2.destroyAllWindows()
