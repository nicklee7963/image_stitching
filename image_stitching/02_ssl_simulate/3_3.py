import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 基礎設定
sift = cv2.SIFT_create(nfeatures=5000) # 設定最大特徵點數
bf = cv2.BFMatcher()

base_height = 45
all_heights = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
key_groups = [50, 70, 90, 95]
sift_inlier_counts = []
image_folder = 'images'

def process_image_sift(path, target_long_side=1200):
    img_bgr = cv2.imread(path)
    if img_bgr is None: return None, None, (0, 0)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    h, w = img_rgb.shape[:2]
    scale = target_long_side / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized_rgb = cv2.resize(img_rgb, (new_w, new_h))
    img_gray = cv2.cvtColor(img_resized_rgb, cv2.COLOR_RGB2GRAY)
    
    return img_gray, img_resized_rgb, (new_w, new_h)

# 讀取基準影像
img0_gray, img0_rgb, (w0, h0) = process_image_sift(os.path.join(image_folder, f'{base_height}.jpg'))
kp0, des0 = sift.detectAndCompute(img0_gray, None)

print(f"🚀 開始 SIFT 對照組分析 (基準特徵點數: {len(kp0)})")

for h in all_heights:
    img1_path = os.path.join(image_folder, f'{h}.jpg')
    img1_gray, img1_rgb, (w1, h1) = process_image_sift(img1_path)
    if img1_gray is None: continue
    
    # --- SIFT 特徵匹配 ---
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    
    # 使用 KNN 進行匹配並套用 Lowe's Ratio Test (0.75)
    matches = bf.knnMatch(des0, des1, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    # 轉換坐標格式以利 RANSAC
    src_pts = np.float32([kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # --- RANSAC 過濾 ---
    num_inliers = 0
    mkpts0_in, mkpts1_in = [], []
    
    if len(src_pts) > 10:
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers_mask = mask.ravel().astype(bool)
            mkpts0_in = src_pts[inliers_mask].reshape(-1, 2)
            mkpts1_in = dst_pts[inliers_mask].reshape(-1, 2)
            num_inliers = len(mkpts0_in)

    sift_inlier_counts.append(num_inliers)
    print(f"📊 SIFT: 45 vs {h} | 原始匹配: {len(good_matches)} | RANSAC 後正確點: {num_inliers}")

    # --- 關鍵組別視覺化 ---
    if h in key_groups:
        combined = np.hstack((img0_rgb, img1_rgb))
        plt.figure(figsize=(20, 10))
        plt.imshow(combined)
        
        display_num = min(num_inliers, 120)
        if num_inliers > 0:
            indices = np.random.choice(num_inliers, display_num, replace=False)
            for idx in indices:
                plt.plot([mkpts0_in[idx, 0], mkpts1_in[idx, 0] + w0], 
                         [mkpts0_in[idx, 1], mkpts1_in[idx, 1]], 
                         color='#FFFF00', alpha=0.7, linewidth=0.8) # 黃色連線區分 SIFT
                plt.scatter([mkpts0_in[idx, 0], mkpts1_in[idx, 0] + w0], 
                            [mkpts0_in[idx, 1], mkpts1_in[idx, 1]], color='blue', s=3)
            
        plt.title(f"SIFT + RANSAC: 45cm vs {h}cm (Inliers: {num_inliers})", fontsize=16)
        plt.axis('off')
        plt.savefig(f'sift_vis_{base_height}_{h}.png', bbox_inches='tight', dpi=200)
        plt.close()

# --- 2. 繪製 SIFT 趨勢圖 ---
plt.figure(figsize=(12, 7))
plt.plot(all_heights, sift_inlier_counts, marker='x', color='#9B59B6', linewidth=3, label='SIFT Inliers')
plt.xlabel('Camera Height (cm)')
plt.ylabel('Number of Correct Matches')
plt.title('SIFT Performance (Traditional Baseline)')
plt.grid(True, linestyle=':')
plt.legend()
plt.savefig('sift_performance_trend.png')
plt.show()
