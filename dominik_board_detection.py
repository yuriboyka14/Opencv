import cv2
import numpy as np

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, gray = cv2.threshold(gray, 80, 230, cv2.THRESH_BINARY)

    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # blur = cv2.bilateralFilter(gray,9,75,75)
    blur = gray

    edges = cv2.Canny(blur, 40, 170)
    dillated_edges = cv2.dilate(edges.copy(), None, iterations = 5)
    dillated_edges = cv2.morphologyEx(dillated_edges, cv2.MORPH_CLOSE, np.ones((8,8),np.uint8))

    return dillated_edges 


def check_cricles(new_crc, old_crc):
    th = 10
 
    if old_crc is not None:
        for index_new, new in enumerate(new_crc[0, :]):
            for index_old, old in enumerate(old_crc[0, :]):
                if new[0] in range(old[0]-th, old[0]+th) and new[1] in range(old[1]-th, old[1]+th):
                    continue
                else:
                    old_crc[0, index_old] = new_crc[0, index_new]

        return old_crc

    else:
        return new_crc

def find_circle(processed_img, last_circles):
    rows = processed_img.shape[0]
    circles = cv2.HoughCircles(processed_img, cv2.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=30,minRadius=10, maxRadius=100)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles = check_cricles(circles, last_circles)
        return circles

    return last_circles

def print_cricles(img, circles):
    if circles is not None:
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = 30 
            cv2.circle(img, center, radius, (187, 206, 125), 3)


def circle_position(circles, cnt):
    grid_return = np.zeros((3,3), dtype=bool)
    bnd_x,bnd_y,bnd_w,bnd_h = cv2.boundingRect(cnt)

    if cnt is not None:
        if circles is not None:
            left_x = bnd_x
            up_y = bnd_y
            down_y = bnd_y+bnd_h
            right_x = bnd_x+bnd_w

            for cr in circles[0, :]:
                (x, y) = (cr[0], cr[1])

                if x < left_x and y < up_y:
                    grid_return[0][0] = "o"

                elif x < left_x and down_y > y >= up_y:
                    grid_return[1][0] = "o"

                elif x < left_x and y > down_y:
                    grid_return[2][0] = "o"

                # second column
                elif right_x > x >= left_x and y < up_y:
                    grid_return[0][1] = "o"

                elif right_x > x >= left_x and down_y > y >= up_y:
                    grid_return[1][1] = "o"

                elif right_x > x >= left_x and y > down_y:
                    grid_return[2][1] = "o"

                # third column
                elif right_x <= x and y < up_y:
                    grid_return[0][2] = "o"

                elif right_x <= x and down_y > y >= up_y:
                    grid_return[1][2] = "o"

                elif right_x <= x and y > down_y:
                    grid_return[2][2] = "o"

                else:
                    continue
    
    return grid_return

def find_print_contour(src, dst, prev_contour):
    contours, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        accuracy = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, accuracy, True)
        
        if len(approx) == 4 and cv2.contourArea(approx) > 15000:   
            cv2.drawContours(dst, [approx], -1, (120,255,155), 10)
            return approx
        
    if prev_contour is not None:
        cv2.drawContours(dst, [prev_contour], -1, (150,255,155), 3)
        return prev_contour


vid = cv2.VideoCapture(0)

contour = None
last_circles = None
grid = None

while True:
    ret, img = vid.read()
    edges = preprocess(img)

    contour = find_print_contour(edges, img, contour)

    last_circles = find_circle(edges, last_circles)
    print_cricles(img, last_circles)

    k = cv2.waitKey(50)

    if k & 0xFF == ord('q'):
        if grid is not None:
            print(grid)
        break

    if k & 0xFF == ord('\r'):
        grid = circle_position(last_circles, contour)
        print(grid)
 
    cv2.imshow('image', img)
    cv2.imshow('edges', edges)
        
vid.release()
cv2.destroyAllWindows()
