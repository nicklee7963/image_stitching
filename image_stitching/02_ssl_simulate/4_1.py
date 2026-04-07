import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from kornia.feature import LoFTR
from kornia.geometry import resize

# 1. 基礎設定與模型載入
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loftr = LoFTR(pretrained='outdoor').to(device).eval()
image_folder = 'images'
target_long_side = 1200 
base_height = 45
# 遍歷所有拍攝間隔
all_test_heights = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

def load_and_preprocess(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None: return None, None, (0, 0)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    h, w = img_rgb.shape[:2]
    scale = target_long_side / max(h, w)
    new_h, new_w = int(h * scale // 16 * 16), int(w * scale // 16 * 16)
    
    img_res = cv2.resize(img_rgb, (new_w, new_h))
    img_gray = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)
    img_torch = torch.from_numpy(img_gray).float()[None, None].to(device) / 255.0
    return img_res, img_torch, (new_w, new_h)

# 讀取基準影像 (45cm)
img0_rgb, img0_torch, (w0, h0) = load_and_process = load_and_preprocess(os.path.join(image_folder, f'{base_height}.jpg'))

print(f"🚀 開始執行全量縫合分析 (基準: {base_height}cm, 設備: {device})")

for h in all_test_heights:
    img1_path = os.path.join(image_folder, f'{h}.jpg')
    img1_rgb, img1_torch, (w1, h1) = load_and_preprocess(img1_path)
    if img1_rgb is None: continue
    
    # --- 步驟 A: LoFTR 特徵匹配 ---
    input_dict = {"image0": img0_torch, "image1": img1_torch}
    with torch.no_grad():
        correspondences = loftr(input_dict)
    
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    
    # --- 步驟 B: 計算 H 矩陣 (使用 RANSAC 篩選) ---
    # 將基準圖 (45cm) 投影至目標高度圖之坐標系
    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
    
    if H is not None:
        # --- 步驟 C: 建立拼接畫布 ---
        # 由於是垂直向上拍攝，基準影像投影後會出現在目標影像下方
        # 畫布高度設定為目標影像高度加上位移量 (估算值)
        vertical_offset = int((h - base_height) * (h1 / 15)) # 根據位移比例動態擴展
        canvas_h = h1 + vertical_offset + 200
        canvas_w = w1
        
        # 執行透視變換 (Warp Perspective)
        # 
        stitched = cv2.warpPerspective(img0_rgb, H, (canvas_w, canvas_h))
        
        # --- 步驟 D: 影像疊加 ---
        # 將目標影像 (較高位置的照片) 覆蓋在畫布頂部
        stitched[0:h1, 0:w1] = img1_rgb
        
        # --- 步驟 E: 儲存與展示 ---
        plt.figure(figsize=(8, 12))
        plt.imshow(stitched)
        inliers = np.sum(mask)
        plt.title(f"Stitching Result: {base_height}cm vs {h}cm (Inliers: {inliers})")
        plt.axis('off')
        
        save_path = f'final_stitch_{base_height}_{h}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"✅ 已完成: {save_path} | 有效匹配數: {inliers}")
    else:
        print(f"❌ 無法計算 {h}cm 之矩陣，跳過該組。")

print("\n✨ 所有高度間隔之縫合成果已處理完畢。")
