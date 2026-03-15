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
IMG_DIR = "../../image/tv_desk"
OUTPUT_DIR = "."
CSV_PATH = os.path.join(OUTPUT_DIR, "loftr_apap_parallax_metrics.csv")

BASE_IMG_NAME = "40.jpg"
base_img_path = os.path.join(IMG_DIR, BASE_IMG_NAME)
base_img = cv2.imread(base_img_path)

if base_img is None:
    print(f"❌ 找不到基準圖片: {base_img_path}")
    exit()

TARGET_IMGS = [45, 50, 55, 60, 65, 70]
results = []

print("🤖 正在載入 LoFTR 模型進入 GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matcher = KF.LoFTR(pretrained='outdoor').to(device).eval()

# ==========================================
# 2. APAP 參數對比設定 (A/B Test)
# ==========================================
# 定義兩種不同的 APAP 參數性格

# ==========================================
# 2. APAP 參數對比設定 (極限 A/B Test)
# ==========================================
APAP_CONFIGS = [
    {
        "name": "Rigid", 
        "cell_size": 50, 
        "sigma_factor": 0.1,   # 參數極大：牽一髮動全身，強迫維持傳統的死板矩陣
        "gamma": 0.1,          # 權重極大：完全不允許局部變形
        "color": (0, 0, 255)   # 紅色標題
    },
    {
        "name": "Extreme_Jelly", 
        "cell_size": 15,       # 網格切超細 (15x15 pixel 一格)，讓它有空間極度扭曲
        "sigma_factor": 0.003, # 參數極小：特徵點只會拉扯自己周圍「極小」範圍的像素
        "gamma": 1e-6,         # 趨近於 0：徹底解除全局形狀的束縛，讓它隨便扭
        "color": (0, 255, 0)   # 綠色標題
    }
]




def normalize_points(pts):
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

    print(f"\n👉 正在測試 LoFTR + APAP 配對: 40 vs {t} ...")
    
    h, w = base_img.shape[:2]
    MAX_DIM = 840
    scale = MAX_DIM / max(h, w)
    if scale < 1.0:
        resize_h, resize_w = int(h * scale), int(w * scale)
    else:
        resize_h, resize_w = h, w
        
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
    
    del t1, t2, out
    torch.cuda.empty_cache()
    
    sort_idx = np.argsort(confidences)[::-1]
    pts1_raw = pts1_raw[sort_idx]
    pts2_raw = pts2_raw[sort_idx]
    confidences = confidences[sort_idx]
    
    pts1_scaled = pts1_raw * np.array([w / new_w, h / new_h])
    pts2_scaled = pts2_raw * np.array([w / new_w, h / new_h])
    
    num_good_matches = len(pts1_scaled)
    status = "Success"
    inliers_count = 0

    if num_good_matches >= 10:
        src_pts = np.float32(pts1_scaled).reshape(-1, 2)
        dst_pts = np.float32(pts2_scaled).reshape(-1, 2)
        
        H_global, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H_global is not None:
            inliers_count = np.sum(mask)
            inlier_src = src_pts[mask.ravel() == 1]
            inlier_dst = dst_pts[mask.ravel() == 1]
            
            MAX_APAP_POINTS = 800
            if len(inlier_src) > MAX_APAP_POINTS:
                inlier_src = inlier_src[:MAX_APAP_POINTS]
                inlier_dst = inlier_dst[:MAX_APAP_POINTS]
            
            h1, w1 = base_img.shape[:2]
            h2, w2 = target_img.shape[:2]
            
            corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners_img2, H_global)
            corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            all_corners = np.concatenate((corners_img1, warped_corners), axis=0)
            
            [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            
            if (xmax - xmin) > 10000 or (ymax - ymin) > 10000:
                print("  ❌ 幾何崩潰：全域透視變形過大。")
                status = "Collapsed (Global H Blowup)"
            else:
                cw, ch = xmax - xmin, ymax - ymin
                tx, ty = -xmin, -ymin
                inlier_src_canvas = inlier_src + np.array([tx, ty])
                
                norm_src, T_src = normalize_points(inlier_src_canvas)
                norm_dst, T_dst = normalize_points(inlier_dst)
                
                N = len(norm_src)
                A = np.zeros((N, 2, 9))
                for i in range(N):
                    x, y = norm_src[i]
                    u, v = norm_dst[i]
                    A[i, 0] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
                    A[i, 1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]

                # 字典用來暫存兩種參數跑出來的畫布
                canvases = {}

                # ==========================================
                # 🌟 APAP 雙核參數執行 (A/B Test)
                # ==========================================
                for config in APAP_CONFIGS:
                    p_name = config["name"]
                    c_size = config["cell_size"]
                    print(f"  ✨ 執行 [{p_name}] 變形 (Cell:{c_size}, Sigma:{config['sigma_factor']}, Gamma:{config['gamma']})")
                    
                    grid_X, grid_Y = np.meshgrid(np.arange(0, cw, c_size), np.arange(0, ch, c_size))
                    mesh_H, mesh_W = grid_X.shape
                    
                    map_x_grid = np.zeros((mesh_H, mesh_W), dtype=np.float32)
                    map_y_grid = np.zeros((mesh_H, mesh_W), dtype=np.float32)
                    
                    sigma = max(cw, ch) * config["sigma_factor"]
                    gamma = config["gamma"]
                    
                    for i in range(mesh_H):
                        for j in range(mesh_W):
                            gx, gy = grid_X[i, j], grid_Y[i, j]
                            dists_sq = (inlier_src_canvas[:, 0] - gx)**2 + (inlier_src_canvas[:, 1] - gy)**2
                            weights = np.maximum(np.exp(-dists_sq / (sigma**2)), gamma)
                            
                            WA = A * weights[:, None, None]
                            WA = WA.reshape(2*N, 9)
                            
                            _, _, Vt = np.linalg.svd(WA)
                            H_norm = Vt[-1].reshape(3, 3)
                            H_inv_local = np.linalg.inv(T_dst).dot(H_norm).dot(T_src)
                            
                            pt = np.array([gx, gy, 1.0])
                            mapped_pt = H_inv_local.dot(pt)
                            mapped_pt /= (mapped_pt[2] + 1e-6)
                            
                            map_x_grid[i, j] = mapped_pt[0]
                            map_y_grid[i, j] = mapped_pt[1]
                    
                    map_x = cv2.resize(map_x_grid, (cw, ch), interpolation=cv2.INTER_LINEAR)
                    map_y = cv2.resize(map_y_grid, (cw, ch), interpolation=cv2.INTER_LINEAR)
                    
                    warped_target = cv2.remap(target_img, map_x, map_y, cv2.INTER_LINEAR)
                    
                    # 影像合成
                    canvas = warped_target.copy()
                    canvas[ty:ty+h1, tx:tx+w1] = base_img
                    
                    # 單獨存檔
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"apap_{p_name}_40_vs_{t}.jpg"), canvas)
                    canvases[p_name] = canvas

                # 製作左右對比圖
                if "Conservative" in canvases and "Aggressive" in canvases:
                    img_c = canvases["Conservative"].copy()
                    img_a = canvases["Aggressive"].copy()
                    
                    # 加上明顯的標題文字
                    cv2.putText(img_c, "Conservative (Global-like)", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, APAP_CONFIGS[0]["color"], 5)
                    cv2.putText(img_a, "Aggressive (Local Warp)", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, APAP_CONFIGS[1]["color"], 5)
                    
                    # 縮小一半並排，避免檔案過大難以開啟
                    scale_down = 0.5
                    img_c_sm = cv2.resize(img_c, (0,0), fx=scale_down, fy=scale_down)
                    img_a_sm = cv2.resize(img_a, (0,0), fx=scale_down, fy=scale_down)
                    
                    comparison = np.hstack((img_c_sm, img_a_sm))
                    comp_path = os.path.join(OUTPUT_DIR, f"A_B_Compare_40_vs_{t}.jpg")
                    cv2.imwrite(comp_path, comparison)
                    print(f"  📸 對比圖已生成: A_B_Compare_40_vs_{t}.jpg")
                
        else:
            status = "Collapsed (Global H Failed)"
    else:
        status = "Failed (Insufficient Matches)"

print("\n🎉 測試完成！請去資料夾查看 A_B_Compare 系列的圖片！")
