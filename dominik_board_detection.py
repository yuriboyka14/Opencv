import cv2
import numpy as np

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, gray = cv2.threshold(gray, 80, 230, cv2.THRESH_BINARY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 170)
    dillated_edges = cv2.dilate(edges.copy(), None, iterations = 5)
    dillated_edges = cv2.morphologyEx(dillated_edges, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    return dillated_edges 


def check_cricles(new_crc, old_crc):
    th = 30
    print("x")
    for new in new_crc[0, :]:
        for old in old_crc[0, :]:
            if new[0] in range(old[0]-th, old[0]+th) or new[1] in range(old[1]-th, old[1]+th):
                continue
            else:
                old = new

    return old_crc

def find_circle(processed_img, output_img, last_circles):
    rows = processed_img.shape[0]
    circles = cv2.HoughCircles(processed_img, cv2.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=30,minRadius=10, maxRadius=100)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print("nowe:")
        print(circles)
        print("stare:")
        print(last_circles)
        if last_circles is not None:
            circles = check_cricles(circles, last_circles)
        return circles

def print_cricles(img, circles):
    if circles is not None:
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(img, center, radius, (187, 206, 125), 3)


def circle_position(circles, cnt):
    grid_return = np.zeros((3,3), dtype=bool)
    bnd_x,bnd_y,bnd_w,bnd_h = cv2.boundingRect(cnt)


    if cnt is not None:
        if circles is not None:
            leftup_x = bnd_x
            rightup_y = bnd_y
            leftdown_y = bnd_x+bnd_h
            rightdown_x = bnd_x+bnd_w

            for cr in circles[0, :]:
                (x, y) = (cr[0], cr[1])

                grid_x = np.zeros((3,3), dtype=bool)
                grid_y = np.zeros((3,3), dtype=bool)

                #grid_x
                if x <= leftup_x:
                    grid_x[:][0] = True
                
                elif x >= leftup_x and x <= rightdown_x:
                    grid_x[:][1] = True
                
                else:
                    grid_x[:][2] = True

                #grid_y
                if y > leftdown_y:
                    grid_y[0][:] = True
                
                elif y <= leftdown_y and y >= rightup_y:
                    grid_y[1][:] = True
                
                else:
                    grid_y[2][:] = True

                grid_x = np.rot90(grid_x)

                #grid
                for t in range(3):
                    for r in range(3):
                        grid_return[t][r] = grid_return[t][r] or (grid_x[t][r] and grid_y[t][r]) 
    
        else:
            print("No circles!")
    else:
        print("No contour!")
    
    return grid_return


vid = cv2.VideoCapture(0)

last_contour = None
last_circles = None
grid = None

while True:
    ret, img = vid.read()
    processed_img = preprocess(img)

    contours, _ = cv2.findContours(processed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        accuracy = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, accuracy, True)
        
        if len(approx) == 4 and cv2.contourArea(approx) > 15000:
            last_contour = approx
            cv2.drawContours(img, [approx], -1, (120,255,155), 10)
        
        elif last_contour is not None:
            cv2.drawContours(img, [last_contour], -1, (150,255,155), 3)

    last_circles = find_circle(processed_img, img, last_circles)
    print_cricles(img, last_circles)

    k = cv2.waitKey(50)

    if k & 0xFF == ord('q'):
        if grid is not None:
            print(grid)
        break

    if k & 0xFF == ord('\r'):
        grid = circle_position(last_circles, last_contour)
        if grid is not None:
            print(grid)
    
    cv2.imshow('image', img)
    cv2.imshow('edges', processed_img)
        
vid.release()
cv2.destroyAllWindows()
