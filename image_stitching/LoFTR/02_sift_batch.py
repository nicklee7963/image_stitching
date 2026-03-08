import cv2
import numpy as np
import os
from pathlib import Path

# --- 設定區域 ---
IMG_DIR = Path("../images/20251205")
SAVE_DIR = Path("./SIFT")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 初始化 SIFT 偵測器
sift = cv2.SIFT_create()

# FLANN 匹配器參數 (SIFT 標準配備)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def process_sift(plant_folder):
    # 1. 辨識圖片
    all_files = list(plant_folder.glob("*.[jJ][pP][gG]")) + list(plant_folder.glob("*.[pP][nN][gG]"))
    img_up_path = next((f for f in all_files if "up" in f.name.lower()), None)
    img_down_path = next((f for f in all_files if "down" in f.name.lower()), None)

    if not img_up_path or not img_down_path:
        return

    # 2. 讀取全尺寸圖片
    img_up = cv2.imread(str(img_up_path))
    img_down = cv2.imread(str(img_down_path))

    if img_up is None or img_down is None:
        return

    # 3. SIFT 特徵提取
    gray_up = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    gray_down = cv2.cvtColor(img_down, cv2.COLOR_BGR2GRAY)
    
    kp1, des1 = sift.detectAndCompute(gray_up, None)
    kp2, des2 = sift.detectAndCompute(gray_down, None)

    if des1 is None or des2 is None:
        print(f"跳過 {plant_folder.name}: 找不到足夠特徵點")
        return

    # 4. 進行匹配與 Lowe's Ratio Test
    matches = flann.knnMatch(des1, des2, k=2)
    
    # 儲存好的匹配點
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 5. RANSAC 過濾
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 使用單應性矩陣或基礎矩陣進行過濾
        M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 1.0)
        matches_mask = mask.ravel().tolist()
        inliers_count = np.sum(matches_mask)
    else:
        matches_mask = None
        inliers_count = 0

    # 6. 繪製結果 (為了預覽，我們將結果拼接後存檔)
    # 注意：因為是全尺寸，vis_img 會非常巨大
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(0, 0, 255),
                       matchesMask=matches_mask,
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    vis_img = cv2.drawMatches(img_up, kp1, img_down, kp2, good_matches, None, **draw_params)

    # 在左上角標註資訊
    h, w = img_up.shape[:2]
    info = [
        f"SIFT Matches: {inliers_count}",
        f"Full Res: {w}x{h}",
        f"Plant: {plant_folder.name}"
    ]
    
    # 因為全尺寸圖片很大，字體要調大一點才看得到
    font_scale = max(1.5, w / 1000)
    thickness = max(2, int(w / 500))
    for i, text in enumerate(info):
        cv2.putText(vis_img, text, (50, int(100 + i * font_scale * 50)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

    # 7. 儲存
    save_path = SAVE_DIR / f"{plant_folder.name}_sift_full.jpg"
    cv2.imwrite(str(save_path), vis_img)
    print(f"[{plant_folder.name}] SIFT 處理完成，Inliers: {inliers_count}")

if __name__ == "__main__":
    plant_folders = sorted(list(IMG_DIR.glob("plant_*")), key=lambda x: int(x.name.split('_')[1]))
    print(f"開始執行 SIFT 全尺寸匹配...")
    for folder in plant_folders:
        process_sift(folder)
    print(f"\n任務完成！結果存於: {SAVE_DIR}")
