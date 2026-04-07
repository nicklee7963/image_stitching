import cv2
import numpy as np
import os
import pandas as pd
import torch
import kornia as K
import kornia.feature as KF

# ==========================================
# 1. 初始化與路徑設定
# ==========================================
IMG_DIR = "../../benchmark/image_simulation/single_translation"
OUTPUT_DIR = "result_simulation/single_translation"
CSV_PATH = os.path.join(OUTPUT_DIR, "loftr_parallax_metrics.csv")

BASE_IMG_NAME = "40.jpg"
base_img_path = os.path.join(IMG_DIR, BASE_IMG_NAME)
base_img = cv2.imread(base_img_path)

if base_img is None:
    print(f"❌ 找不到基準圖片: {base_img_path}")
    exit()

TARGET_IMGS = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
results = []

print("🤖 正在載入 LoFTR 模型進入 GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matcher = KF.LoFTR(pretrained='outdoor').to(device).eval()

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
    
    h, w = base_img.shape[:2]
    
    # 🌟 關鍵修復 1：限制最大解析度，防止矩陣爆炸 OOM
    MAX_DIM = 840
    scale = MAX_DIM / max(h, w)
    if scale < 1.0:
        resize_h, resize_w = int(h * scale), int(w * scale)
    else:
        resize_h, resize_w = h, w
        
    # LoFTR 要求長寬必須是 8 的倍數
    new_h, new_w = (resize_h // 8) * 8, (resize_w // 8) * 8
    
    img1_resized = cv2.resize(base_img, (new_w, new_h))
    img2_resized = cv2.resize(target_img, (new_w, new_h))
    
    t1 = K.image_to_tensor(cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY), False).float().to(device) / 255.
    t2 = K.image_to_tensor(cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY), False).float().to(device) / 255.
    
    with torch.no_grad():
        out = matcher({"image0": t1, "image1": t2})
        
    pts1_raw = out['keypoints0'].cpu().numpy()
    pts2_raw = out['keypoints1'].cpu().numpy()
    confidences = out['confidence'].cpu().numpy()
    
    # 🌟 關鍵修復 2：每一輪算完立刻手動釋放 GPU 記憶體，避免碎片化累積
    del t1, t2, out
    torch.cuda.empty_cache()
    
    sort_idx = np.argsort(confidences)[::-1]
    pts1_raw = pts1_raw[sort_idx]
    pts2_raw = pts2_raw[sort_idx]
    confidences = confidences[sort_idx]
    
    # 🌟 將算出來的座標，按照原始尺寸的比例放大回去
    pts1_scaled = pts1_raw * np.array([w / new_w, h / new_h])
    pts2_scaled = pts2_raw * np.array([w / new_w, h / new_h])
    
    num_good_matches = len(pts1_scaled)
    inliers_count = 0
    inlier_ratio = 0.0
    status = "Success"
    
    kp1_mock = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=5) for p in pts1_scaled]
    kp2_mock = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=5) for p in pts2_scaled]
    mock_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=float(1 - confidences[i])) for i in range(num_good_matches)]
    
    match_img = cv2.drawMatches(base_img, kp1_mock, target_img, kp2_mock, mock_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"match_40_vs_{t}.jpg"), match_img)

    if num_good_matches >= 4:
        src_pts = np.float32(pts1_scaled).reshape(-1, 1, 2)
        dst_pts = np.float32(pts2_scaled).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is not None:
            inliers_count = np.sum(mask)
            inlier_ratio = inliers_count / num_good_matches if num_good_matches > 0 else 0
            
            h1, w1 = base_img.shape[:2]
            h2, w2 = target_img.shape[:2]
            
            corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            try:
                warped_corners = cv2.perspectiveTransform(corners_img2, H)
                corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
                all_corners = np.concatenate((corners_img1, warped_corners), axis=0)
                
                [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
                [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
                
                translation_dist = [-xmin, -ymin]
                H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
                
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
        
    results.append({
        "Base_Image": BASE_IMG_NAME,
        "Target_Image": target_name,
        "Parallax_Angle": t - 40,
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
print(" 📊 LoFTR 視差崩潰壓力測試報告")
print("="*60)
print(df[['Target_Image', 'Parallax_Angle', 'Good_Matches', 'RANSAC_Inliers', 'Inlier_Ratio', 'Status']].to_string(index=False))
print("="*60)
print(f"📁 量化數據已儲存至：{CSV_PATH}")
print(f"📁 視覺化配對圖與拼接圖已儲存至目前資料夾")
