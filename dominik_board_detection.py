import cv2
import numpy as np

vid = cv2.VideoCapture(0)
kernel2 = np.ones((5,5),np.uint8)
while True:
    ret, img = vid.read()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, gray = cv2.threshold(gray, 80, 230, cv2.THRESH_BINARY)

    kernel = 5
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 90, 150)
    dillated_edges = cv2.dilate(edges.copy(), None, iterations = 4)
    dillated_edges = cv2.morphologyEx(dillated_edges, cv2.MORPH_CLOSE, kernel2)

    col_edges = cv2.cvtColor(dillated_edges, cv2.COLOR_GRAY2RGB)

    contours, _ = cv2.findContours(dillated_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        accuracy = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, accuracy, True)
        if len(approx) == 4:
            cv2.drawContours(col_edges, approx, -1, (120,255,155), 10)
            print(cv2.contourArea(approx))
    cv2.imshow('image', col_edges)
    cv2.imshow('edges', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
