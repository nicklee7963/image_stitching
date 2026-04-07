import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from kornia.feature import LoFTR
from kornia.geometry import resize

# 1. 基礎設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loftr = LoFTR(pretrained='outdoor').to(device).eval()
sift = cv2.SIFT_create(nfeatures=5000)
bf = cv2.BFMatcher()
image_folder = 'images'
target_long_side = 1200 # 使用你指定的高解析度

def load_and_process(path, brightness_factor=1.0):
    img_bgr = cv2.imread(path)
    if img_bgr is None: return None, None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 縮放至 1200px (16的倍數)
    h, w = img_rgb.shape[:2]
    scale = target_long_side / max(h, w)
    new_h, new_w = int(h * scale // 16 * 16), int(w * scale // 16 * 16)
    img_res = cv2.resize(img_rgb, (new_w, new_h))
    
    # 調整亮度並防止溢位
    img_adj = np.clip(img_res.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
    img_gray = cv2.cvtColor(img_adj, cv2.COLOR_RGB2GRAY)
    return img_adj, img_gray, (new_w, new_h)

def get_loftr_inliers(img0_gray, img1_gray):
    t0 = torch.from_numpy(img0_gray).float()[None, None].to(device) / 255.0
    t1 = torch.from_numpy(img1_gray).float()[None, None].to(device) / 255.0
    with torch.no_grad():
        matches = loftr({"image0": t0, "image1": t1})
    mkpts0, mkpts1 = matches['keypoints0'].cpu().numpy(), matches['keypoints1'].cpu().numpy()
    if len(mkpts0) > 10:
        _, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
        return mkpts0[mask.ravel().astype(bool)], mkpts1[mask.ravel().astype(bool)]
    return np.array([]), np.array([])

def get_sift_inliers(img0_gray, img1_gray):
    kp0, des0 = sift.detectAndCompute(img0_gray, None)
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    matches = bf.knnMatch(des0, des1, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    src = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 2)
    if len(src) > 10:
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return src[mask.ravel().astype(bool)], dst[mask.ravel().astype(bool)]
    return np.array([]), np.array([])

# --- 執行實驗 ---
print(f"🚀 開始分析光影下降率 (45 vs 70cm, 設備: {device})")
img45_c, img45_g, (w_ref, h_ref) = load_and_process(os.path.join(image_folder, '45.jpg'), 1.0)
img70_n_c, img70_n_g, _ = load_and_process(os.path.join(image_folder, '70.jpg'), 1.0)
img70_d_c, img70_d_g, _ = load_and_process(os.path.join(image_folder, '70.jpg'), 0.6) # 變暗 40%

# 數據跑分
loftr_norm_in, _ = get_loftr_inliers(img45_g, img70_n_g)
loftr_dark_in, loftr_dark_pts1 = get_loftr_inliers(img45_g, img70_d_g)
sift_norm_in, _ = get_sift_inliers(img45_g, img70_n_g)
sift_dark_in, _ = get_sift_inliers(img45_g, img70_d_g)

# 計算下降百分比
def calc_drop(norm, dark):
    drop = ((norm - dark) / norm) * 100 if norm > 0 else 0
    retention = (dark / norm) * 100 if norm > 0 else 0
    return round(drop, 2), round(retention, 2)

l_drop, l_ret = calc_drop(len(loftr_norm_in), len(loftr_dark_in))
s_drop, s_ret = calc_drop(len(sift_norm_in), len(sift_dark_in))

print(f"\n[LoFTR] 正常: {len(loftr_norm_in)} | 變暗: {len(loftr_dark_in)} | 下降: {l_drop}% | 留存: {l_ret}%")
print(f"[SIFT ] 正常: {len(sift_norm_in)} | 變暗: {len(sift_dark_in)} | 下養: {s_drop}% | 留存: {s_ret}%")

# --- 繪製下降率對比圖 (Word 3.4 重點圖表) ---
plt.figure(figsize=(10, 6))
algos = ['LoFTR', 'SIFT']
drops = [l_drop, s_drop]
colors = ['#2ecc71', '#9b59b6']

bars = plt.bar(algos, drops, color=colors, edgecolor='black', width=0.5)
plt.ylabel('Percentage Decrease (%)', fontsize=12)
plt.title('Performance Drop after 40% Brightness Reduction', fontsize=14)
plt.ylim(0, max(drops) + 20)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.savefig('lighting_drop_analysis.png', dpi=150)
plt.show()

print("\n✨ 分析完成！請檢查 'lighting_drop_analysis.png'。")
