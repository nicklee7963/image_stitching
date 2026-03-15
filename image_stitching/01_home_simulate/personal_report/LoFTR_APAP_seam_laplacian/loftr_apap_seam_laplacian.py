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
# 🌟 確保路徑退回兩層：personal_report -> 01_home_simulate -> image
IMG_DIR = "../../image/tv_desk"
OUTPUT_DIR = "."
# 🌟 更新 CSV 檔名
CSV_PATH = os.path.join(OUTPUT_DIR, "loftr_apap_final_metrics.csv")

BASE_IMG_NAME = "40.jpg"
base_img_path = os.path.join(IMG_DIR, BASE_IMG_NAME)
base_img = cv2.imread(base_img_path)

if base_img is None:
    print(f"❌ 找不到基準圖片: {base_img_path} (請確認路徑層級是否正確)")
    exit()

TARGET_IMGS = [45, 50, 55, 60, 65, 70]
results = []

# ==========================================
# 🌟 核心改動：手動指定 40.jpg 中彌勒佛神像的絕對區域
# 格式為 (ymin, ymax, xmin, xmax)，請根據原圖解析度手動估計
# 在 2048x1536 的解析度下，手動預估：
DYNAMIC_OBJECT_CROP = (1200, 1900, 1200, 1900)
# ==========================================

print("🤖 正在載入 LoFTR 模型進入 GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matcher = KF.LoFTR(pretrained='outdoor').to(device).eval()

# ==========================================
# 2. 輔助函式 (APAP, Robust Seam, Laplacian, Crop, Dynamic Mask)
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

def find_robust_seam_mask(mask1, mask2):
    """最穩固的 Voronoi 距離接縫法，保證只有一條乾淨的切割線"""
    dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
    dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)
    seam_mask = (dist1 > dist2).astype(np.float32)
    # 稍微模糊邊緣，避免鋸齒
    seam_mask = cv2.GaussianBlur(seam_mask, (5, 5), 0)
    return seam_mask

def create_dynamic_object_canvas_mask(src_shape, canvas_shape, crop, map_x, map_y):
    """在畫布上生成指定動態物件的遮罩，並縮放回原本解析度"""
    h_src, w_src = src_shape[:2]
    ch, cw = canvas_shape[:2]
    
    # 1. 先在原圖尺寸上生成一個框
    raw_mask_src = np.zeros((h_src, w_src), dtype=np.uint8)
    ymin, ymax, xmin, xmax = crop
    raw_mask_src[ymin:ymax, xmin:xmax] = 255
    
    # 2. 利用 APAP 的重對應 (Remap)把這個框扭曲到全景畫布上
    # map_x/map_y 是把畫布座標對應回原圖的坐標系，我們需要一個簡單的貼地過程
    # 這裡我們換個思維：我們已經有 APAP 生成的 twisted base image (沒有，base沒 twisted)
    # OK, APAP map 是針對 Target 的。基準圖 (40.jpg) 是直接貼在tx,ty偏移量上。
    
    # 更正：基準圖 (40.jpg) 只是被平移tx,ty，沒有 twisted
    # 所以動態物件的遮罩只是平移
    dynamic_mask_canvas = np.zeros((ch, cw), dtype=np.uint8)
    
    # 這裡還不能平移，因為我們還不知道 tx,ty
    return raw_mask_src # 先傳原圖遮罩出去

def laplacian_blend(img1, img2, mask, levels=5):
    """拉普拉斯金字塔融合"""
    h, w = img1.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    
    i1 = cv2.copyMakeBorder(img1, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT).astype(np.float32)
    i2 = cv2.copyMakeBorder(img2, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT).astype(np.float32)
    m = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT).astype(np.float32)
    
    gp_m, gp_i1, gp_i2 = [m], [i1], [i2]
    for i in range(levels):
        m, i1, i2 = cv2.pyrDown(m), cv2.pyrDown(i1), cv2.pyrDown(i2)
        gp_m.append(m); gp_i1.append(i1); gp_i2.append(i2)
        
    lp_i1, lp_i2 = [gp_i1[-1]], [gp_i2[-1]]
    for i in range(levels, 0, -1):
        size = (gp_i1[i-1].shape[1], gp_i1[i-1].shape[0])
        lp_i1.append(cv2.subtract(gp_i1[i-1], cv2.pyrUp(gp_i1[i], dstsize=size)))
        lp_i2.append(cv2.subtract(gp_i2[i-1], cv2.pyrUp(gp_i2[i], dstsize=size)))
        
    LS = []
    for i, (l1, l2) in enumerate(zip(lp_i1, lp_i2)):
        current_mask = gp_m[levels - i]
        if len(current_mask.shape) == 2: current_mask = cv2.merge([current_mask]*3)
        LS.append(l1 * current_mask + l2 * (1.0 - current_mask))
        
    ls_ = LS[0]
    for i in range(1, levels + 1):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.add(cv2.pyrUp(ls_, dstsize=size), LS[i])
        
    return np.clip(ls_[:h, :w], 0, 255).astype(np.uint8)

def crop_black_border(img):
    """自動裁切全景圖周圍的黑色死區"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img
    max_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_cnt)
    return img[y:y+h, x:x+w]

# ==========================================
# 3. 視差漸進測試迴圈
# ==========================================
for t in TARGET_IMGS:
    target_name = f"{t}.jpg"
    target_img_path = os.path.join(IMG_DIR, target_name)
    target_img = cv2.imread(target_img_path)
    
    if target_img is None: continue

    print(f"\n👉 正在生成終極縫合照片: 40 vs {t} ...")
    
    # --- LoFTR 密集特徵配對 ---
    h, w = base_img.shape[:2]
    scale = 840 / max(h, w)
    resize_h, resize_w = int(h * scale), int(w * scale) if scale < 1.0 else (h, w)
    new_h, new_w = (resize_h // 8) * 8, (resize_w // 8) * 8
    img1_resized = cv2.resize(base_img, (new_w, new_h))
    img2_resized = cv2.resize(target_img, (new_w, new_h))
    t1 = K.image_to_tensor(cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY), False).float().to(device) / 255.
    t2 = K.image_to_tensor(cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY), False).float().to(device) / 255.
    with torch.no_grad(): out = matcher({"image0": t1, "image1": t2})
    
    pts1_scaled = out['keypoints0'].cpu().numpy()[np.argsort(out['confidence'].cpu().numpy())[::-1]] * np.array([w / new_w, h / new_h])
    pts2_scaled = out['keypoints1'].cpu().numpy()[np.argsort(out['confidence'].cpu().numpy())[::-1]] * np.array([w / new_w, h / new_h])
    del t1, t2, out
    torch.cuda.empty_cache()
    
    if len(pts1_scaled) >= 10:
        H_global, mask = cv2.findHomography(np.float32(pts2_scaled).reshape(-1, 2), np.float32(pts1_scaled).reshape(-1, 2), cv2.RANSAC, 5.0)
        
        if H_global is not None:
            # 🌟 使用激進的 APAP 變形參數以對齊 3D 視差
            inliers_count = np.sum(mask)
            inlier_src = pts1_scaled[mask.ravel() == 1][:800]
            inlier_dst = pts2_scaled[mask.ravel() == 1][:800]
            
            h1, w1 = base_img.shape[:2]
            h2, w2 = target_img.shape[:2]
            corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners2, H_global)
            corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            all_corners = np.concatenate((corners1, warped_corners), axis=0)
            [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            
            if (xmax - xmin) < 10000 and (ymax - ymin) < 10000:
                cw, ch, tx, ty = xmax - xmin, ymax - ymin, -xmin, -ymin
                
                # --- APAP 網格變形 (Jelly Mode) ---
                print("  ✨ 執行 APAP 網格變形...")
                inlier_src_canvas = inlier_src + np.array([tx, ty])
                CELL_SIZE, sigma, gamma = 25, max(cw, ch) * 0.015, 0.0001
                grid_X, grid_Y = np.meshgrid(np.arange(0, cw, CELL_SIZE), np.arange(0, ch, CELL_SIZE))
                mesh_H, mesh_W = grid_X.shape
                norm_src, T_src = normalize_points(inlier_src_canvas)
                norm_dst, T_dst = normalize_points(inlier_dst)
                A = np.zeros((len(norm_src), 2, 9))
                for i, ((x, y), (u, v)) in enumerate(zip(norm_src, norm_dst)):
                    A[i, 0] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
                    A[i, 1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
                
                map_x_grid, map_y_grid = np.zeros((mesh_H, mesh_W), dtype=np.float32), np.zeros((mesh_H, mesh_W), dtype=np.float32)
                for i in range(mesh_H):
                    for j in range(mesh_W):
                        gx, gy = grid_X[i, j], grid_Y[i, j]
                        weights = np.maximum(np.exp(-((inlier_src_canvas[:, 0] - gx)**2 + (inlier_src_canvas[:, 1] - gy)**2) / (sigma**2)), gamma)
                        WA = (A * weights[:, None, None]).reshape(-1, 9)
                        _, _, Vt = np.linalg.svd(WA)
                        H_inv_local = np.linalg.inv(T_dst).dot(Vt[-1].reshape(3, 3)).dot(T_src)
                        pt = np.array([gx, gy, 1.0])
                        mapped_pt = H_inv_local.dot(pt)
                        map_x_grid[i, j] = mapped_pt[0] / (mapped_pt[2] + 1e-6)
                        map_y_grid[i, j] = mapped_pt[1] / (mapped_pt[2] + 1e-6)
                
                map_x = cv2.resize(map_x_grid, (cw, ch), interpolation=cv2.INTER_LINEAR)
                map_y = cv2.resize(map_y_grid, (cw, ch), interpolation=cv2.INTER_LINEAR)
                warped_target = cv2.remap(target_img, map_x, map_y, cv2.INTER_LINEAR)
                
                # --- 準備融合畫布 ---
                canvas_target = warped_target.copy()
                mask_target = cv2.remap(np.ones((h2, w2), dtype=np.uint8)*255, map_x, map_y, cv2.INTER_NEAREST)
                canvas_base = np.zeros((ch, cw, 3), dtype=np.uint8)
                canvas_base[ty:ty+h1, tx:tx+w1] = base_img
                mask_base = np.zeros((ch, cw), dtype=np.uint8)
                mask_base[ty:ty+h1, tx:tx+w1] = 255
                
                # ==========================================
                # 🌟 核心改動：生成人為指定的動態物件保留遮罩
                # ==========================================
                print(f"  ✨ 建立人為指定保留遮罩 (區域: {DYNAMIC_OBJECT_CROP})...")
                # 1. 取得原圖上的遮罩
                dynamic_mask_src = np.zeros((h1, w1), dtype=np.uint8)
                ymin, ymax, xmin, xmax = DYNAMIC_OBJECT_CROP
                dynamic_mask_src[ymin:ymax, xmin:xmax] = 255
                
                # 2. 將遮罩平移到全景畫布位置
                dynamic_mask_canvas = np.zeros((ch, cw), dtype=np.float32)
                dynamic_mask_canvas[ty:ty+h1, tx:tx+w1] = (dynamic_mask_src / 255.0)
                # 3. 給這個框一個柔和的邊緣過渡 (GaussianBlur)，避免僵硬切線
                dynamic_mask_canvas = cv2.GaussianBlur(dynamic_mask_canvas, (101, 101), 0)
                
                # --- Smart Seam & Laplacian Blending ---
                print("  ✨ 計算智慧接縫並進行拉普拉斯金字塔融合...")
                seam_mask = find_robust_seam_mask(mask_base, mask_target)
                
                # ==========================================
                # 🌟 核心改動：將人為指定的遮罩強制融入 Seam Mask
                # 權重設定：在指定保留區 (dynamic_mask_canvas)，強制讓遮罩趨近 1.0 (保留 img1 / base)
                # ==========================================
                # seam_mask 決定了 img1 的權重。我們希望指定區的 img1 權重為 1
                # 因此，我們用 maximum：誰大就聽誰的
                seam_mask = np.maximum(seam_mask, dynamic_mask_canvas)
                seam_mask = np.clip(seam_mask, 0.0, 1.0) # 確保數值在 0~1
                
                # 執行金字塔融合
                final_stitched = laplacian_blend(canvas_base, canvas_target, seam_mask, levels=5)
                
                # --- 自動裁切黑色死區 ---
                print("  ✨ 自動裁切全景圖...")
                final_cropped = crop_black_border(final_stitched)
                
                # 儲存最終乾淨的全景圖
                final_path = os.path.join(OUTPUT_DIR, f"LoFTR_APAP_Seam_Final_Guided_40_vs_{t}.jpg")
                cv2.imwrite(final_path, final_cropped)
                print(f"  💾 【完全體全景圖】已儲存: {final_path}")

                # 供你對比用的三格對比圖 (保留原本功能，讓你確認紅線狀態與融合效果)
                # 為了Debug，把指定的Guided Mask範圍用紅色框框畫出來
                final_stitched_debug = final_stitched.copy()
                ymin_c, ymax_c, xmin_c, xmax_c = ymin + ty, ymax + ty, xmin + tx, xmax + tx
                cv2.rectangle(final_stitched_debug, (xmin_c, ymin_c), (xmax_c, ymax_c), (0, 0, 255), 10)
                
                seam_line_viz = cv2.Canny((seam_mask * 255).astype(np.uint8), 100, 200)
                final_with_line = final_stitched.copy()
                final_with_line[seam_line_viz > 0] = [0, 0, 255] # 畫上紅線
                hard_overlay = canvas_target.copy()
                hard_overlay[mask_base > 0] = canvas_base[mask_base > 0]
                
                # 生成對比圖 (縮小以便檢視)
                sc = 0.4
                img_hard_sm = cv2.resize(hard_overlay, (0,0), fx=sc, fy=sc)
                img_line_sm = cv2.resize(final_with_line, (0,0), fx=sc, fy=sc)
                img_blend_sm = cv2.resize(final_stitched, (0,0), fx=sc, fy=sc)
                img_guided_sm = cv2.resize(final_stitched_debug, (0,0), fx=sc, fy=sc)
                
                cv2.putText(img_hard_sm, "Hard Overlay", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                cv2.putText(img_line_sm, "Guided Seam (Red Line)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
                cv2.putText(img_blend_sm, "Laplacian Blend", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                cv2.putText(img_guided_sm, "Guided Zone (Red Box)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
                
                # 四格對比
                comparison_top = np.hstack((img_hard_sm, img_line_sm))
                comparison_bottom = np.hstack((img_guided_sm, img_blend_sm))
                comparison = np.vstack((comparison_top, comparison_bottom))
                
                comp_path = os.path.join(OUTPUT_DIR, f"Debug_Guided_Compare_40_vs_{t}.jpg")
                cv2.imwrite(comp_path, comparison)
                
                results.append({"Base": BASE_IMG_NAME, "Target": target_name, "Status": "Success"})
            else:
                results.append({"Base": BASE_IMG_NAME, "Target": target_name, "Status": "Collapsed"})
        else:
            results.append({"Base": BASE_IMG_NAME, "Target": target_name, "Status": "Global H Failed"})

print("\n🎉 全部測試完成！請在資料夾中尋找 \"LoFTR_APAP_Seam_Final_Guided\" 開頭的乾淨全景圖！")
