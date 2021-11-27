import cv2
import numpy as np

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, gray = cv2.threshold(gray, 80, 230, cv2.THRESH_BINARY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 170)
    dillated_edges = cv2.dilate(edges.copy(), None, iterations = 3)
    dillated_edges = cv2.morphologyEx(dillated_edges, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    return dillated_edges 

def find_circle(processed_img, output_img, last_circles):
    rows = processed_img.shape[0]
    circles = cv2.HoughCircles(processed_img, cv2.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=30,minRadius=10, maxRadius=100)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        last_circles = circles

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(output_img, center, radius, (187, 206, 125), 3)

    elif last_circles is not None:
        for i in last_circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(output_img, center, radius, (187, 206, 125), 3)

    return circles
            

vid = cv2.VideoCapture(0)
last_contour = None
last_circles = None

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


    find_circle(processed_img, img, last_circles)
    
    k = cv2.waitKey(2)

    if k & 0xFF == ord('q'):
        break

    cv2.imshow('image', img)
    cv2.imshow('edges', processed_img)
        
vid.release()
cv2.destroyAllWindows()
