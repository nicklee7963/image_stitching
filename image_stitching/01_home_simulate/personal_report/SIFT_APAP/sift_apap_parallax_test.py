import cv2
import numpy as np
import os
import pandas as pd

# ==========================================
# 1. 初始化與路徑設定
# ==========================================
IMG_DIR = "../../image/tv_desk"
OUTPUT_DIR = "."
CSV_PATH = os.path.join(OUTPUT_DIR, "sift_apap_parallax_metrics.csv")

BASE_IMG_NAME = "40.jpg"
base_img_path = os.path.join(IMG_DIR, BASE_IMG_NAME)
base_img = cv2.imread(base_img_path)

if base_img is None:
    print(f"❌ 找不到基準圖片: {base_img_path}")
    exit()

TARGET_IMGS = [45, 50, 55, 60, 65, 70]
results = []
sift = cv2.SIFT_create()

print(f"📸 提取基準圖 {BASE_IMG_NAME} 特徵中...")
kp1, des1 = sift.detectAndCompute(base_img, None)

# ==========================================
# 2. APAP (Moving DLT) 輔助函式
# ==========================================
def normalize_points(pts):
    """將特徵點座標正規化，避免 SVD 矩陣計算時數值爆炸"""
    mean = np.mean(pts, axis=0)
    dist = np.mean(np.sqrt(np.sum((pts - mean)**2, axis=1)))
    scale = np.sqrt(2) / (dist + 1e-6)
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])
    pts_norm = np.dot(T, np.vstack((pts.T, np.ones(len(pts)))))
    return pts_norm[:2].T, T

# ==========================================
# 3. 視差漸進測試迴圈
# ==========================================
for t in TARGET_IMGS:
    target_name = f"{t}.jpg"
    target_img_path = os.path.join(IMG_DIR, target_name)
    target_img = cv2.imread(target_img_path)
    
    if target_img is None:
        continue

    print(f"\n👉 正在測試 APAP 配對: 40 vs {t} ...")
    kp2, des2 = sift.detectAndCompute(target_img, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    num_good_matches = len(good_matches)
    status = "Success"
    inliers_count = 0

    if num_good_matches >= 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        
        # 1. 計算 Global Homography 取得全景圖畫布邊界，並過濾 Inliers
        H_global, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H_global is not None:
            inliers_count = np.sum(mask)
            inlier_src = src_pts[mask.ravel() == 1]
            inlier_dst = dst_pts[mask.ravel() == 1]
            
            h1, w1 = base_img.shape[:2]
            h2, w2 = target_img.shape[:2]
            
            corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners_img2, H_global)
            corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            all_corners = np.concatenate((corners_img1, warped_corners), axis=0)
            
            [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            
            # 若發生極端崩潰，跳過此圖
            if (xmax - xmin) > 10000 or (ymax - ymin) > 10000:
                print("  ❌ 幾何崩潰：全域透視變形過大。")
                status = "Collapsed (Global H Blowup)"
            else:
                cw, ch = xmax - xmin, ymax - ymin
                tx, ty = -xmin, -ymin
                
                # ==========================================
                # 🌟 APAP 核心：動態網格局部變形 (Moving DLT)
                # ==========================================
                print(f"  ✨ 啟動 APAP 網格局部變形計算 (Canvas: {cw}x{ch})...")
                
                # 調整基準點座標到新畫布上
                inlier_src_canvas = inlier_src + np.array([tx, ty])
                
                # 網格切分 (數值越小越精細，但算越久。50是很好的平衡)
                CELL_SIZE = 50
                grid_X, grid_Y = np.meshgrid(np.arange(0, cw, CELL_SIZE), np.arange(0, ch, CELL_SIZE))
                mesh_H, mesh_W = grid_X.shape
                
                # SVD 數值正規化
                norm_src, T_src = normalize_points(inlier_src_canvas)
                norm_dst, T_dst = normalize_points(inlier_dst)
                
                N = len(norm_src)
                A = np.zeros((N, 2, 9))
                for i in range(N):
                    x, y = norm_src[i]
                    u, v = norm_dst[i]
                    A[i, 0] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
                    A[i, 1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
                
                map_x_grid = np.zeros((mesh_H, mesh_W), dtype=np.float32)
                map_y_grid = np.zeros((mesh_H, mesh_W), dtype=np.float32)
                
                # APAP 參數
                sigma = max(cw, ch) * 0.05  # 高斯權重衰減半徑
                gamma = 0.005               # 避免矩陣奇異的基礎權重 (Global H)
                
                # 計算每個網格點專屬的局部矩陣
                for i in range(mesh_H):
                    for j in range(mesh_W):
                        gx, gy = grid_X[i, j], grid_Y[i, j]
                        
                        # 計算該網格點到所有特徵點的距離權重
                        dists_sq = (inlier_src_canvas[:, 0] - gx)**2 + (inlier_src_canvas[:, 1] - gy)**2
                        weights = np.maximum(np.exp(-dists_sq / (sigma**2)), gamma)
                        
                        # 權重乘上 DLT 矩陣
                        WA = A * weights[:, None, None]
                        WA = WA.reshape(2*N, 9)
                        
                        # SVD 求解局部 H
                        _, _, Vt = np.linalg.svd(WA)
                        H_norm = Vt[-1].reshape(3, 3)
                        
                        # 逆正規化取得真實座標系的局部 H_inv
                        H_inv_local = np.linalg.inv(T_dst).dot(H_norm).dot(T_src)
                        
                        # 利用局部 H 將畫布網格反推回目標圖的像素位置
                        pt = np.array([gx, gy, 1.0])
                        mapped_pt = H_inv_local.dot(pt)
                        mapped_pt /= (mapped_pt[2] + 1e-6)
                        
                        map_x_grid[i, j] = mapped_pt[0]
                        map_y_grid[i, j] = mapped_pt[1]
                
                # 將粗糙網格平滑插值成全畫布尺寸的 map
                map_x = cv2.resize(map_x_grid, (cw, ch), interpolation=cv2.INTER_LINEAR)
                map_y = cv2.resize(map_y_grid, (cw, ch), interpolation=cv2.INTER_LINEAR)
                
                # 將目標圖無縫重對應(Remap)到畫布上
                warped_target = cv2.remap(target_img, map_x, map_y, cv2.INTER_LINEAR)
                
                # 合成：基準圖覆蓋上去
  # 合成：基準圖直接貼到畫布對應位置
                canvas = warped_target.copy()
                canvas[ty:ty+h1, tx:tx+w1] = base_img              
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"apap_stitch_40_vs_{t}.jpg"), canvas)
                print(f"  ✅ APAP 拼接成功，Inliers: {inliers_count}")
                
        else:
            status = "Collapsed (Global H Failed)"
    else:
        status = "Failed (Insufficient Matches)"
        
    results.append({
        "Base_Image": BASE_IMG_NAME,
        "Target_Image": target_name,
        "Parallax_Angle": t - 40,
        "Good_Matches": num_good_matches,
        "RANSAC_Inliers": inliers_count,
        "Status": status
    })

# ==========================================
# 4. 儲存數據
# ==========================================
df = pd.DataFrame(results)
df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

print("\n" + "="*60)
print(" 📊 SIFT + APAP 網格局部變形測試報告")
print("="*60)
print(df.to_string(index=False))
print("="*60)
