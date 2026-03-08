import cv2
import numpy as np
import sys
import os

def rotate_image(image_path, angle):
    # 1. 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤：找不到圖片 {image_path}")
        return

    # 2. 處理旋轉邏輯
    if angle in [90, 180, 270, -90]:
        # 使用 OpenCV 內建的快速翻轉 (不失真、不產生黑邊)
        if angle == 90:
            out = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            out = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270 or angle == -90:
            out = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # 處理任意角度 (會產生黑邊補償以確保圖不被切到)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 計算旋轉後的畫布大小，避免裁切
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]
        out = cv2.warpAffine(img, M, (nW, nH))

    # 3. 覆蓋原檔
    cv2.imwrite(image_path, out)
    print(f"成功！已旋轉 {angle} 度並覆蓋：{image_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使用方式: python3 rotate_tool.py [圖片路徑] [旋轉角度]")
        print("範例: python3 rotate_tool.py ../images/20251205/plant_1/down.jpg 180")
    else:
        path = sys.argv[1]
        deg = float(sys.argv[2])
        rotate_image(path, deg)
