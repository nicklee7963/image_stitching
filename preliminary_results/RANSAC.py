import cv2
import numpy as np
import os

def compare_sift_ransac(img1_path, img2_path, ransac_threshold=5.0):
    print("--- SIFT 特徵匹配與 RANSAC 對比測試 ---")
    
    # 1. 讀取圖片
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("錯誤：找不到圖片，請確認路徑。")
        return
        
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2. SIFT 偵測與描述子計算
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 3. KNN 匹配與 Lowe's Ratio Test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"-> 基礎過濾後 (Before RANSAC) 匹配數量: {len(good_matches)}")

    # 4. 繪製 RANSAC 前的結果 (包含亂配的點)
    img_before = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 165, 255) # 橘色線條代表初步匹配
    )
    cv2.imwrite("01_Before_RANSAC.jpg", img_before)
    print("-> 已儲存 RANSAC 前的匹配圖：01_Before_RANSAC.jpg")

    # 5. 執行 RANSAC
    if len(good_matches) >= 4:
        # 提取匹配點座標
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 計算單應性矩陣並取得 mask (1=Inlier, 0=Outlier)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        matchesMask = mask.ravel().tolist()
        
        inlier_count = sum(matchesMask)
        inlier_ratio = inlier_count / len(good_matches)
        
        print(f"-> RANSAC 過濾後 (After RANSAC) 匹配數量: {inlier_count}")
        print(f"-> RANSAC 保留率 (Inlier Ratio): {inlier_ratio:.2%}")

        # 6. 繪製 RANSAC 後的結果 (只畫 Inliers)
        draw_params = dict(
            matchColor=(0, 255, 0),       # 綠色線條代表正確匹配
            singlePointColor=None,
            matchesMask=matchesMask,      # 透過 mask 只畫出 Inliers
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        img_after = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
        cv2.imwrite("02_Before_And_After_RANSAC_Diff.jpg", img_after)
        print("-> 已儲存 RANSAC 後的匹配圖：02_Before_And_After_RANSAC_Diff.jpg")
        
    else:
        print("-> 匹配點少於 4 個，無法執行 RANSAC。")

if __name__ == "__main__":
    # 請替換成你的照片檔名
    compare_sift_ransac("image1.jpg", "image2.jpg", ransac_threshold=5.0)

