import cv2
import torch
import kornia
from kornia.feature import LoFTR
import numpy as np
import os

# ==========================================
# 核心工具：繪圖與存檔 (修改版：強制縮放至相同大小)
# ==========================================
def draw_and_save(img1_origin, img2_origin, pts1, pts2, filename, info_text):
    # 複製一份以免汙染原圖
    img1_vis = img1_origin.copy()
    img2_vis = img2_origin.copy()
    
    # 取得原始尺寸
    h1, w1 = img1_vis.shape[:2]
    h2, w2 = img2_vis.shape[:2]

    # --- 修改重點：強制將圖2縮放成圖1的大小 ---
    # 目標尺寸：以圖1為標準 (或是你可以自訂一個標準)
    target_w, target_h = w1, h1
    
    # 縮放圖2
    img2_resized = cv2.resize(img2_vis, (target_w, target_h))
    
    # 計算縮放比例 (用來修正點的座標)
    scale_x = target_w / w2
    scale_y = target_h / h2

    # 垂直堆疊 (現在兩張圖大小一模一樣了，可以直接疊)
    vis = np.vstack((img1_vis, img2_resized))
    
    # --- 畫線邏輯 ---
    count = 0
    if pts1 is not None and len(pts1) > 0:
        count = len(pts1)
        for p1, p2 in zip(pts1, pts2):
            # p1 是在圖1上，不用動
            pt1 = (int(p1[0]), int(p1[1]))
            
            # p2 是在原始圖2上，必須乘上縮放比例，再加上圖1的高度(因為疊在下面)
            pt2_x = p2[0] * scale_x
            pt2_y = p2[1] * scale_y
            pt2 = (int(pt2_x), int(pt2_y + h1))
            
            # 畫線與點
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1)      # 綠線
            cv2.circle(vis, pt1, 4, (0, 0, 255), -1)     # 紅點 (上圖)
            cv2.circle(vis, pt2, 4, (0, 0, 255), -1)     # 紅點 (下圖)

    # 寫上資訊
    label = f"{info_text} | Count: {count}"
    # 字體稍微縮小一點以免擋住畫面，位置動態調整
    cv2.putText(vis, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # 存檔
    cv2.imwrite(filename, vis)
    print(f"已儲存: {filename} (匹配數: {count})")

# ==========================================
# 任務 1: SIFT (原圖大小)
# ==========================================
def run_sift_task(img1, img2):
    print("\n[任務 1] 正在執行 SIFT (原圖解析度)...")
    
    h, w = img1.shape[:2]
    print(f"\n[任務 1] 正在執行 SIFT (原圖解析度: 寬 {w} x 高 {h})...")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
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
        
        if len(good) > 0:
            p1 = np.float32([kp1[m.queryIdx].pt for m in good])
            p2 = np.float32([kp2[m.trainIdx].pt for m in good])
            
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
        crop_h1 = int(h1 * 0.5)
        crop_y1 = h1 - crop_h1
        input_img1 = img1[crop_y1:, :] 
        
        crop_h2 = int(h2 * 0.5)
        input_img2 = img2[:crop_h2, :] 
        
        offset_x1, offset_y1 = 0, crop_y1
        offset_x2, offset_y2 = 0, 0
    else:
        input_img1 = img1
        input_img2 = img2
        offset_x1, offset_y1 = 0, 0
        offset_x2, offset_y2 = 0, 0

    # --- 步驟 B: 縮放至 LoFTR 推論大小 ---
    infer_w, infer_h = resolution
    src_h1, src_w1 = input_img1.shape[:2]
    src_h2, src_w2 = input_img2.shape[:2]
    
    img1_resized = cv2.resize(input_img1, (infer_w, infer_h))
    img2_resized = cv2.resize(input_img2, (infer_w, infer_h))
    
    tensor1 = kornia.image_to_tensor(cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    tensor2 = kornia.image_to_tensor(cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained='outdoor').to(device)
    
    with torch.no_grad():
        res = matcher({"image0": tensor1.to(device), "image1": tensor2.to(device)})

    mkpts0 = res['keypoints0'].cpu().numpy()
    mkpts1 = res['keypoints1'].cpu().numpy()
    conf = res['confidence'].cpu().numpy()

    valid = conf > 0.3
    pts1 = mkpts0[valid]
    pts2 = mkpts1[valid]
    
    # --- 步驟 D: 座標還原 ---
    if len(pts1) > 0:
        scale_x1 = src_w1 / infer_w
        scale_y1 = src_h1 / infer_h
        scale_x2 = src_w2 / infer_w
        scale_y2 = src_h2 / infer_h
        
        pts1[:, 0] *= scale_x1
        pts1[:, 1] *= scale_y1
        pts2[:, 0] *= scale_x2
        pts2[:, 1] *= scale_y2
        
        pts1[:, 0] += offset_x1
        pts1[:, 1] += offset_y1
        pts2[:, 0] += offset_x2
        pts2[:, 1] += offset_y2

        if len(pts1) >= 4:
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            if mask is not None:
                matches_mask = mask.ravel().tolist()
                pts1 = pts1[np.array(matches_mask) == 1]
                pts2 = pts2[np.array(matches_mask) == 1]
    
    roi_tag = "ROI" if use_roi else "NoROI"
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
        
        run_sift_task(img1, img2)
        
        resolutions = [
            (640, 480),   
            (800, 600),   
            (1280, 960)   
        ]
        res_names = ["01_Small", "02_Medium", "03_High"]
        
        for res, name in zip(resolutions, res_names):
            run_loftr_task(img1, img2, res, f"NoROI_{name}", use_roi=False)
            
        for res, name in zip(resolutions, res_names):
            run_loftr_task(img1, img2, res, f"WithROI_{name}", use_roi=True)
            
        print("\n=== 全部完成！ ===")
    else:
        print("錯誤：找不到 image1.jpg 或 image2.jpg")
