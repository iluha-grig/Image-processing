import numpy as np
import cv2 as cv
import os


EPS = 30
EPS2 = 20
BIN_THRESHOLD = 55
DEPTH_THRESHOLD = 21000
RAD_THRESHOLD = np.pi / 2


def show_release(img, img_name):
    cv.imshow(img_name, img)
    cv.waitKey(0)


def find_points(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_gray, BIN_THRESHOLD, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv.contourArea(x))
    hull = cv.convexHull(contours, returnPoints=False)
    defects = cv.convexityDefects(contours, hull)
    list_end = []
    list_start = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i][0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
        if angle <= RAD_THRESHOLD and d > DEPTH_THRESHOLD:  # angle less than 90 degree, treat as fingers
            cv.circle(img, far, 4, [0, 0, 255], -1)
            list_start.append(start)
            list_end.append(end)
    list_res = []
    list_end_tmp = list_end.copy()
    for first in list_start:
        for second in list_end:
            if np.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) < EPS:
                list_res.append(((first[0] + second[0]) // 2, (first[1] + second[1]) // 2))
                list_end_tmp.remove(second)
                break
        else:
            list_res.append(first)
    if len(list_end_tmp) != 1:
        raise Exception('Error execute')
    else:
        list_res.append(list_end_tmp[0])
    for point in list_res:
        cv.circle(img, point, 4, [0, 0, 255], -1)
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i][0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        if angle <= RAD_THRESHOLD and d > DEPTH_THRESHOLD:
            for point in list_res:
                if np.sqrt((point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2) < EPS2:
                    cv.line(img, point, far, [0, 0, 255], 3, -1)
                if np.sqrt((point[0] - end[0]) ** 2 + (point[1] - end[1]) ** 2) < EPS2:
                    cv.line(img, point, far, [0, 0, 255], 3, -1)


if __name__ == '__main__':
    for name in os.listdir('training'):
        img = cv.imread('training/' + name)
        find_points(img)
        show_release(img, name)

# 091.tif, 109.tif errors
