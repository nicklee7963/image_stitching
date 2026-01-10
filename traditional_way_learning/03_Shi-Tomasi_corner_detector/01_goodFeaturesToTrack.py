import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('chessboard.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray,5,0.01,50)
corners = np.int8(corners)

for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)

# 2. 設定顯示內容
plt.imshow(img)

# 3. 真正秀出來
plt.show()
