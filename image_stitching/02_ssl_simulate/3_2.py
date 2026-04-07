import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from kornia.feature import LoFTR
from kornia.geometry import resize

# 1. 基礎環境設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loftr = LoFTR(pretrained='outdoor').to(device).eval()

base_height = 45
all_heights = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
key_groups = [50, 70, 90, 95]  # 你指定的四組關鍵視覺化對象
inlier_counts = []             # 儲存過濾後的正確匹配數
image_folder = 'images'

def process_image(path, target_long_side=1200):
    img_bgr = cv2.imread(path)
    if img_bgr is None: return None, None, (0, 0)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 計算縮放比例 (長邊 1200px, 且需為 16 的倍數)
    h, w = img_rgb.shape[:2]
    scale = target_long_side / max(h, w)
    new_h, new_w = int(h * scale // 16 * 16), int(w * scale // 16 * 16)
    
    img_resized_rgb = cv2.resize(img_rgb, (new_w, new_h))
    # 轉為灰階 Tensor 供模型運算
    img_gray = cv2.cvtColor(img_resized_rgb, cv2.COLOR_RGB2GRAY)
    img_torch = torch.from_numpy(img_gray).float()[None, None].to(device) / 255.0
    
    return img_torch, img_resized_rgb, (new_w, new_h)

# 讀取基準影像 (45cm)
img0_torch, img0_rgb, (w0, h0) = process_image(os.path.join(image_folder, f'{base_height}.jpg'))

print(f"🚀 開始高解析度分析 (解析度: {w0}x{h0}, 設備: {device})")

for h in all_heights:
    img1_path = os.path.join(image_folder, f'{h}.jpg')
    img1_torch, img1_rgb, (w1, h1) = process_image(img1_path)
    if img1_torch is None: continue
    
    # --- LoFTR 特徵匹配 ---
    input_dict = {"image0": img0_torch, "image1": img1_torch}
    with torch.no_grad():
        correspondences = loftr(input_dict)
    
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    
    # --- RANSAC 過濾誤匹配 (Outlier Removal) ---
    if len(mkpts0) > 10:
        # 使用 RANSAC 找出單應性矩陣，threshold 設為 5.0 像素
        H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
        inliers_mask = mask.ravel().astype(bool)
        
        # 僅保留正確的匹配點 (Inliers)
        mkpts0_in = mkpts0[inliers_mask]
        mkpts1_in = mkpts1[inliers_mask]
        num_inliers = len(mkpts0_in)
    else:
        mkpts0_in, mkpts1_in = mkpts0, mkpts1
        num_inliers = len(mkpts0)

    inlier_counts.append(num_inliers)
    print(f"📊 高度 {base_height} vs {h} | 原始點數: {len(mkpts0)} | RANSAC 後正確點: {num_inliers}")

    # --- 關鍵組別彩色視覺化 (只畫正確的點) ---
    if h in key_groups:
        combined = np.hstack((img0_rgb, img1_rgb))
        plt.figure(figsize=(20, 10))
        plt.imshow(combined)
        
        # 隨機抽取 120 條正確連線以利展示
        display_num = min(num_inliers, 120)
        if num_inliers > 0:
            indices = np.random.choice(num_inliers, display_num, replace=False)
            for idx in indices:
                # 綠色連線 (#00FF00) 與 紅色端點
                plt.plot([mkpts0_in[idx, 0], mkpts1_in[idx, 0] + w0], 
                         [mkpts0_in[idx, 1], mkpts1_in[idx, 1]], 
                         color='#00FF00', alpha=0.7, linewidth=0.8)
                plt.scatter([mkpts0_in[idx, 0], mkpts1_in[idx, 0] + w0], 
                            [mkpts0_in[idx, 1], mkpts1_in[idx, 1]], 
                            color='#FF0000', s=3)
            
        plt.title(f"LoFTR + RANSAC: {base_height}cm vs {h}cm (Inliers: {num_inliers})", fontsize=16)
        plt.axis('off')
        plt.savefig(f'report_vis_{base_height}_{h}.png', bbox_inches='tight', dpi=200)
        plt.close()

# --- 2. 繪製全間隔「正確匹配數」趨勢圖 ---
plt.figure(figsize=(12, 7))
plt.plot(all_heights, inlier_counts, marker='o', color='#E67E22', linewidth=3, markersize=10, label='Inlier Matches')
plt.fill_between(all_heights, inlier_counts, color='#E67E22', alpha=0.1)
plt.xlabel('Camera Height (cm)', fontsize=12)
plt.ylabel('Number of Correct Matches (Inliers)', fontsize=12)
plt.title('Correct Feature Matching Trend (1200px + RANSAC)', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.savefig('loftr_ransac_trend.png')
plt.show()

print("\n✨ 實驗分析完成！圖表已根據 RANSAC 過濾結果重新產出。")
