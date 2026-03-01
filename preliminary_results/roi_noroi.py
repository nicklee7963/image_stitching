import cv2
import torch
import kornia
from kornia.feature import LoFTR
import numpy as np
import os

# ==========================================
# 1. LoFTR 提取 (含 ROI 邏輯)
# ==========================================
def get_loftr_matches(img1, img2, use_roi=False, resolution=(1280, 960)):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # --- 裁切 ROI 或 使用全圖 ---
    if use_roi:
        # 圖1取下半，圖2取上半 (針對垂直拼接的重疊區)
        crop_h1 = int(h1 * 0.5)
        crop_y1 = h1 - crop_h1
        input1 = img1[crop_y1:, :] 
        
        crop_h2 = int(h2 * 0.5)
        input2 = img2[:crop_h2, :] 
        
        offset_x1, offset_y1 = 0, crop_y1
        offset_x2, offset_y2 = 0, 0
    else:
        input1 = img1
        input2 = img2
        offset_x1, offset_y1 = 0, 0
        offset_x2, offset_y2 = 0, 0

    # --- 縮放與推論 ---
    infer_w, infer_h = resolution
    src_h1, src_w1 = input1.shape[:2]
    src_h2, src_w2 = input2.shape[:2]
    
    img1_resized = cv2.resize(input1, (infer_w, infer_h))
    img2_resized = cv2.resize(input2, (infer_w, infer_h))
    
    tensor1 = kornia.image_to_tensor(cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    tensor2 = kornia.image_to_tensor(cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained='outdoor').to(device)
    
    with torch.no_grad():
        res = matcher({"image0": tensor1.to(device), "image1": tensor2.to(device)})

    mkpts0 = res['keypoints0'].cpu().numpy()
    mkpts1 = res['keypoints1'].cpu().numpy()
    conf = res['confidence'].cpu().numpy()

    # 信心度過濾
    valid = conf > 0.3
    pts1 = mkpts0[valid]
    pts2 = mkpts1[valid]
    
    # --- 座標還原 ---
    if len(pts1) > 0:
        scale_x1 = src_w1 / infer_w
        scale_y1 = src_h1 / infer_h
        scale_x2 = src_w2 / infer_w
        scale_y2 = src_h2 / infer_h
        
        pts1[:, 0] *= scale_x1
        pts1[:, 1] *= scale_y1
        pts2[:, 0] *= scale_x2
        pts2[:, 1] *= scale_y2
        
        # 加上 ROI 偏移量
        pts1[:, 0] += offset_x1
        pts1[:, 1] += offset_y1
        pts2[:, 0] += offset_x2
        pts2[:, 1] += offset_y2
        
    return pts1, pts2

# ==========================================
# 2. RANSAC 過濾器 (獨立使用)
# ==========================================
def filter_matches_ransac(pts1, pts2, thresh=5.0):
    if len(pts1) < 4:
        return pts1, pts2 # 點太少，無法 RANSAC，直接回傳
    
    # 計算單應性矩陣來找 Outliers
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, thresh)
    
    if mask is None:
        return pts1, pts2
        
    matches_mask = mask.ravel().tolist()
    
    # 只保留 Inliers
    pts1_clean = pts1[np.array(matches_mask) == 1]
    pts2_clean = pts2[np.array(matches_mask) == 1]
    
    return pts1_clean, pts2_clean

# ==========================================
# 3. 去除重複點 (Deduplication)
# ==========================================
def remove_duplicates(base_p1, base_p2, new_p1, new_p2, threshold=5.0):
    """
    base: 通常是前景 (ROI) 點，我們想保留的
    new: 通常是背景 (NoROI) 點，如果跟 base 太近，我們就刪掉
    """
    if len(base_p1) == 0: return new_p1, new_p2
    if len(new_p1) == 0: return [], []

    keep_indices = []
    
    # 對每一個新的點 (背景點)
    for i in range(len(new_p1)):
        # 計算它跟所有 base 點 (前景點) 的距離
        # 這裡只比較圖1的座標 (p1) 就足夠判斷是否是同一個特徵了
        diff = base_p1 - new_p1[i]
        dist = np.linalg.norm(diff, axis=1)
        
        # 如果跟任何一個 base 點的距離小於 threshold，視為重複，不保留
        if np.min(dist) > threshold:
            keep_indices.append(i)
            
    return new_p1[keep_indices], new_p2[keep_indices]

# ==========================================
# 主流程
# ==========================================
def run_clean_merge_pipeline(img1_path, img2_path):
    print(">>> 執行: 獨立 RANSAC + 去重複合併...")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # 1. 取得原始點
    print("   [1/4] 提取特徵...")
    raw_bg_p1, raw_bg_p2 = get_loftr_matches(img1, img2, use_roi=False) # 藍色
    raw_fg_p1, raw_fg_p2 = get_loftr_matches(img1, img2, use_roi=True)  # 紅色
    print(f"         原始數量 -> 背景: {len(raw_bg_p1)}, 前景: {len(raw_fg_p1)}")

    # 2. 獨立 RANSAC (關鍵步驟！)
    # 我們分別過濾，這樣背景不會殺死前景，前景也不會殺死背景
    print("   [2/4] 獨立 RANSAC 過濾...")
    clean_bg_p1, clean_bg_p2 = filter_matches_ransac(raw_bg_p1, raw_bg_p2)
    clean_fg_p1, clean_fg_p2 = filter_matches_ransac(raw_fg_p1, raw_fg_p2)
    print(f"         RANSAC後 -> 背景: {len(clean_bg_p1)}, 前景: {len(clean_fg_p1)}")

    # 3. 去除重複
    # 策略：保留前景(紅)，如果背景(藍)有點跟前景重疊，刪除背景的那點
    print("   [3/4] 去除重複點...")
    unique_bg_p1, unique_bg_p2 = remove_duplicates(clean_fg_p1, clean_fg_p2, clean_bg_p1, clean_bg_p2)
    print(f"         去重後背景剩餘: {len(unique_bg_p1)} (刪除了 {len(clean_bg_p1) - len(unique_bg_p1)} 個重複點)")

    # 4. 繪圖
    print("   [4/4] 繪製最終結果...")
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_w = max(w1, w2)
    
    vis_img1 = img1.copy()
    vis_img2 = img2.copy()
    if w1 < target_w: vis_img1 = cv2.copyMakeBorder(vis_img1, 0, 0, 0, target_w-w1, cv2.BORDER_CONSTANT)
    if w2 < target_w: vis_img2 = cv2.copyMakeBorder(vis_img2, 0, 0, 0, target_w-w2, cv2.BORDER_CONSTANT)
    
    vis = np.vstack((vis_img1, vis_img2))
    
    # 畫背景 (藍色) - Unique Background
    for i in range(len(unique_bg_p1)):
        pt1 = (int(unique_bg_p1[i][0]), int(unique_bg_p1[i][1]))
        pt2 = (int(unique_bg_p2[i][0]), int(unique_bg_p2[i][1] + vis_img1.shape[0]))
        cv2.line(vis, pt1, pt2, (255, 0, 0), 1) 
        cv2.circle(vis, pt1, 3, (255, 0, 0), -1)

    # 畫前景 (紅色) - Foreground
    for i in range(len(clean_fg_p1)):
        pt1 = (int(clean_fg_p1[i][0]), int(clean_fg_p1[i][1]))
        pt2 = (int(clean_fg_p2[i][0]), int(clean_fg_p2[i][1] + vis_img1.shape[0]))
        cv2.line(vis, pt1, pt2, (0, 0, 255), 1)
        cv2.circle(vis, pt1, 3, (0, 0, 255), -1)

    # 統計資訊
    total = len(unique_bg_p1) + len(clean_fg_p1)
    cv2.putText(vis, f"Method: Dual-RANSAC + Deduplication", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(vis, f"Blue(BG): {len(unique_bg_p1)} | Red(FG): {len(clean_fg_p1)} | Total: {total}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imwrite("Final_Clean_Merge.jpg", vis)
    print(">>> 完成！已儲存: Final_Clean_Merge.jpg")

if __name__ == "__main__":
    run_clean_merge_pipeline("image1.jpg", "image2.jpg")
