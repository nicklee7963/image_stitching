import cv2
import numpy as np
import os
import pandas as pd

# ==========================================
# 1. 初始化與路徑設定
# ==========================================
# 假設此腳本執行於 01_home_simulate/personal_report/SIFT/
IMG_DIR = "../../benchmark/image_simulation/multiple_translation"
OUTPUT_DIR = "result_simulation/multiple_translation"
CSV_PATH = os.path.join(OUTPUT_DIR, "sift_parallax_metrics.csv")

BASE_IMG_NAME = "40.jpg"
base_img_path = os.path.join(IMG_DIR, BASE_IMG_NAME)
base_img = cv2.imread(base_img_path)

if base_img is None:
    print(f"❌ 找不到基準圖片: {base_img_path}")
    exit()

# 測試的目標圖片列表
TARGET_IMGS = list(range(45, 105, 5))
results = []

sift = cv2.SIFT_create()

# 提取基準圖特徵
print(f"📸 提取基準圖 {BASE_IMG_NAME} 特徵中...")
kp1, des1 = sift.detectAndCompute(base_img, None)

# ==========================================
# 2. 視差漸進測試迴圈
# ==========================================
for t in TARGET_IMGS:
    target_name = f"{t}.jpg"
    target_img_path = os.path.join(IMG_DIR, target_name)
    target_img = cv2.imread(target_img_path)
    
    if target_img is None:
        print(f"⚠️ 找不到目標圖片: {target_img_path}，略過。")
        continue

    print(f"\n👉 正在測試配對: 40 vs {t} ...")
    
    # 提取目標圖特徵
    kp2, des2 = sift.detectAndCompute(target_img, None)
    
    # FLANN 特徵配對
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    num_good_matches = len(good_matches)
    inliers_count = 0
    inlier_ratio = 0.0
    status = "Success"
    
    # 繪製 Matching 圖
    match_img = cv2.drawMatches(base_img, kp1, target_img, kp2, good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"match_40_vs_{t}.jpg"), match_img)

    if num_good_matches >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 計算單應性矩陣 (從 目標圖 變換到 基準圖)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is not None:
            inliers_count = np.sum(mask)
            inlier_ratio = inliers_count / num_good_matches if num_good_matches > 0 else 0
            
            h1, w1 = base_img.shape[:2]
            h2, w2 = target_img.shape[:2]
            
            # 計算變換後的邊界
            corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            try:
                warped_corners = cv2.perspectiveTransform(corners_img2, H)
                corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
                all_corners = np.concatenate((corners_img1, warped_corners), axis=0)
                
                [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
                [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
                
                translation_dist = [-xmin, -ymin]
                H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
                
                # 防呆機制：若矩陣崩潰導致畫布尺寸無限大，直接標記為崩潰
                if (xmax - xmin) > 10000 or (ymax - ymin) > 10000:
                    status = "Collapsed (Matrix Blowup)"
                    print("  ❌ 幾何崩潰：透視變形過大，無法生成全景圖。")
                else:
                    warped_img2 = cv2.warpPerspective(target_img, H_translation.dot(H), (xmax - xmin, ymax - ymin))
                    warped_img2[translation_dist[1]:h1 + translation_dist[1], translation_dist[0]:w1 + translation_dist[0]] = base_img
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"stitch_40_vs_{t}.jpg"), warped_img2)
                    print(f"  ✅ 拼接成功，Inliers: {inliers_count}")
                    
            except Exception as e:
                status = f"Collapsed (Transform Error)"
                print(f"  ❌ 幾何崩潰：計算邊界時發生錯誤。")
        else:
            status = "Collapsed (H matrix is None)"
            print("  ❌ 幾何崩潰：RANSAC 無法收斂算出單應性矩陣。")
    else:
        status = "Failed (Insufficient Matches)"
        print(f"  ❌ 配對失敗：好特徵點不足 ({num_good_matches} < 4)")
        
    # 紀錄量化數值
    results.append({
        "Base_Image": BASE_IMG_NAME,
        "Target_Image": target_name,
        "Parallax_Angle": t - 40,
        "Total_Keypoints_Base": len(kp1),
        "Total_Keypoints_Target": len(kp2),
        "Good_Matches": num_good_matches,
        "RANSAC_Inliers": inliers_count,
        "Inlier_Ratio": round(inlier_ratio, 4),
        "Status": status
    })

# ==========================================
# 3. 儲存數據與生成報告
# ==========================================
df = pd.DataFrame(results)
df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

print("\n" + "="*60)
print(" 📊 SIFT 視差崩潰壓力測試報告")
print("="*60)
print(df[['Target_Image', 'Parallax_Angle', 'Good_Matches', 'RANSAC_Inliers', 'Inlier_Ratio', 'Status']].to_string(index=False))
print("="*60)
print(f"📁 量化數據已儲存至：{CSV_PATH}")
print(f"📁 視覺化配對圖與拼接圖已儲存至目前資料夾")
