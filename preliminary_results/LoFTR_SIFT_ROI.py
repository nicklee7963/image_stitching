import cv2
import torch
import kornia
from kornia.feature import LoFTR
import numpy as np
import os

# ==========================================
# 核心工具：繪圖與存檔 (含自動防呆補黑邊)
# ==========================================
def draw_and_save(img1_origin, img2_origin, pts1, pts2, filename, info_text):
    # 複製一份以免汙染原圖
    img1 = img1_origin.copy()
    img2 = img2_origin.copy()
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 1. 自動統一寬度 (以最寬的為準)
    target_w = max(w1, w2)
    
    if w1 < target_w:
        pad = target_w - w1
        img1 = cv2.copyMakeBorder(img1, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
    if w2 < target_w:
        pad = target_w - w2
        img2 = cv2.copyMakeBorder(img2, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0,0,0))

    # 2. 垂直堆疊
    vis = np.vstack((img1, img2))
    
    # 3. 畫線
    # pts1 在上圖 (y 不變)
    # pts2 在下圖 (y 要加上上圖的高度 h1)
    # 注意：如果 pts1, pts2 是空陣列，zip 會自動跳過
    count = 0
    if pts1 is not None and len(pts1) > 0:
        count = len(pts1)
        for p1, p2 in zip(pts1, pts2):
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1] + h1))
            
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1)      # 綠線
            cv2.circle(vis, pt1, 4, (0, 0, 255), -1)     # 紅點
            cv2.circle(vis, pt2, 4, (0, 0, 255), -1)     # 紅點

    # 4. 寫上資訊 (左上角黃色大字)
    label = f"{info_text} | Count: {count}"
    cv2.putText(vis, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

    # 5. 存檔
    cv2.imwrite(filename, vis)
    print(f"已儲存: {filename} (匹配數: {count})")

# ==========================================
# 任務 1: SIFT (原圖大小)
# ==========================================
def run_sift_task(img1, img2):
    print("\n[任務 1] 正在執行 SIFT (原圖解析度)...")
    
    # 轉灰階
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # SIFT 偵測
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    pts1, pts2 = [], []
    
    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        
        p1 = np.float32([kp1[m.queryIdx].pt for m in good])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        # RANSAC 過濾
        if len(p1) >= 4:
            H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            if mask is not None:
                matches_mask = mask.ravel().tolist()
                pts1 = p1[np.array(matches_mask) == 1]
                pts2 = p2[np.array(matches_mask) == 1]

    draw_and_save(img1, img2, pts1, pts2, "01_SIFT_Original.jpg", "SIFT (Original)")

# ==========================================
# 任務 2 & 3: LoFTR (通用函式)
# ==========================================
def run_loftr_task(img1, img2, resolution, res_name, use_roi):
    mode_str = "ROI" if use_roi else "Full"
    print(f"\n[任務 LoFTR] 模式: {mode_str} | 解析度: {res_name} {resolution}...")
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # --- 步驟 A: 準備圖片 (是否裁切 ROI) ---
    if use_roi:
        # 模擬重疊區：圖1取下半，圖2取上半
        crop_h1 = int(h1 * 0.5)
        crop_y1 = h1 - crop_h1
        input_img1 = img1[crop_y1:, :] # 圖1 底部
        
        crop_h2 = int(h2 * 0.5)
        input_img2 = img2[:crop_h2, :] # 圖2 頂部
        
        # 記錄 offset 以便最後還原座標
        offset_x1, offset_y1 = 0, crop_y1
        offset_x2, offset_y2 = 0, 0
    else:
        # 不裁切，用整張圖
        input_img1 = img1
        input_img2 = img2
        offset_x1, offset_y1 = 0, 0
        offset_x2, offset_y2 = 0, 0

    # --- 步驟 B: 縮放至 LoFTR 推論大小 ---
    infer_w, infer_h = resolution
    
    # 記錄縮放前的尺寸 (為了還原座標)
    src_h1, src_w1 = input_img1.shape[:2]
    src_h2, src_w2 = input_img2.shape[:2]
    
    img1_resized = cv2.resize(input_img1, (infer_w, infer_h))
    img2_resized = cv2.resize(input_img2, (infer_w, infer_h))
    
    # 轉 Tensor
    tensor1 = kornia.image_to_tensor(cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    tensor2 = kornia.image_to_tensor(cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    
    # --- 步驟 C: 執行 LoFTR ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained='outdoor').to(device)
    
    with torch.no_grad():
        res = matcher({"image0": tensor1.to(device), "image1": tensor2.to(device)})

    mkpts0 = res['keypoints0'].cpu().numpy()
    mkpts1 = res['keypoints1'].cpu().numpy()
    conf = res['confidence'].cpu().numpy()

    # 過濾 (信心度 > 0.3)
    valid = conf > 0.3
    pts1 = mkpts0[valid]
    pts2 = mkpts1[valid]
    
    # --- 步驟 D: 座標還原 (兩段式還原) ---
    if len(pts1) > 0:
        # 1. 從「推論解析度」還原回「輸入圖片(可能是ROI)」尺寸
        scale_x1 = src_w1 / infer_w
        scale_y1 = src_h1 / infer_h
        scale_x2 = src_w2 / infer_w
        scale_y2 = src_h2 / infer_h
        
        pts1[:, 0] *= scale_x1
        pts1[:, 1] *= scale_y1
        pts2[:, 0] *= scale_x2
        pts2[:, 1] *= scale_y2
        
        # 2. 如果有 ROI，加上裁切偏移量，還原回「原圖」座標
        pts1[:, 0] += offset_x1
        pts1[:, 1] += offset_y1
        pts2[:, 0] += offset_x2
        pts2[:, 1] += offset_y2

        # RANSAC 過濾
        if len(pts1) >= 4:
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            if mask is not None:
                matches_mask = mask.ravel().tolist()
                pts1 = pts1[np.array(matches_mask) == 1]
                pts2 = pts2[np.array(matches_mask) == 1]
    
    # 存檔 (檔名自動編號)
    # 為了排序方便，我們設定檔名格式
    # ROI 狀態
    roi_tag = "ROI" if use_roi else "NoROI"
    # 解析度標籤
    filename = f"LoFTR_{roi_tag}_{res_name}.jpg"
    
    label_text = f"LoFTR {roi_tag} ({resolution[0]}x{resolution[1]})"
    draw_and_save(img1, img2, pts1, pts2, filename, label_text)


# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    img1_path = "image1.jpg"
    img2_path = "image2.jpg"
    
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # 1. 執行 SIFT
        run_sift_task(img1, img2)
        
        # 設定三種解析度 (寬, 高)
        resolutions = [
            (640, 480),   # Small
            (800, 600),   # Medium
            (1280, 960)   # High (但不是火力全開)
        ]
        res_names = ["01_Small", "02_Medium", "03_High"]
        
        # 2. 執行 LoFTR (No ROI) - 3 種大小
        for res, name in zip(resolutions, res_names):
            run_loftr_task(img1, img2, res, f"NoROI_{name}", use_roi=False)
            
        # 3. 執行 LoFTR (With ROI) - 3 種大小
        for res, name in zip(resolutions, res_names):
            run_loftr_task(img1, img2, res, f"WithROI_{name}", use_roi=True)
            
        print("\n=== 全部完成！請查看產生的 7 張圖片 ===")
    else:
        print("錯誤：找不到 image1.jpg 或 image2.jpg")
