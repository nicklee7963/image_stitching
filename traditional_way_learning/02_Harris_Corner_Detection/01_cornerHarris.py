import numpy as np
import cv2 as cv

filename = 'chessboard.png'
img = cv.imread(filename)

# harris 只看亮度的變化 轉gray可以節省2/3的運算量
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 需要小數點 沒有會捨棄
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 3, 0.04)

dst = cv.dilate(dst, None)

img[dst>0.01*dst.max()] = [0, 0, 255]

cv.imshow('dst', img)
if cv.waitKey(0) & 0xff ==27:  # press esc to exit
    cv.destroyAllWindows()

