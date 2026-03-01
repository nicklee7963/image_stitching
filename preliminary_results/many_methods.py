import cv2
import torch
import kornia
from kornia.feature import LoFTR
import numpy as np
import os

# ==========================================
# 工具函式：繪製 ROI 框與匹配線
# ==========================================
def draw_matches_with_roi(img1, img2, pts1, pts2, roi1_coords, roi2_coords, label, color=(0, 255, 0)):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 建立並排畫布
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
    # 畫 ROI 框 (如果有的話)
    if roi1_coords:
        x, y, w, h = roi1_coords
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 3) # 紅框
        cv2.putText(vis, "ROI", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if roi2_coords:
        x, y, w, h = roi2_coords
        # 右圖的 x 要加上 w1
        cv2.rectangle(vis, (x+w1, y), (x+w1+w, y+h), (0, 0, 255), 3) # 紅框
        cv2.putText(vis, "ROI", (x+w1, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 畫匹配線
    for p1, p2 in zip(pts1, pts2):
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0] + w1), int(p2[1]))
        cv2.line(vis, pt1, pt2, color, 1)
        cv2.circle(vis, pt1, 3, color, -1)
        cv2.circle(vis, pt2, 3, color, -1)

    # 加上文字
    cv2.putText(vis, f"{label}: {len(pts1)} matches", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    return vis

# ==========================================
# 工具函式：單應性矩陣縫合 (Simple Stitch)
# ==========================================
def simple_stitch(img1, img2, H, filename):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 估算畫布大小 (簡化版：假設垂直拼接)
    canvas_h = h1 + h2
    canvas_w = max(w1, w2)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # 貼上 img1
    canvas[:h1, :w1] = img1
    
    # 變形 img2 並貼上
    warped_img2 = cv2.warpPerspective(img2, H, (canvas_w, canvas_h))
    
    # 簡單遮罩融合 (非黑處覆蓋)
    mask = (canvas == 0)
    canvas[mask] = warped_img2[mask]
    
    # 簡單裁切黑邊
    rows = np.any(canvas != 0, axis=(1, 2))
    cols = np.any(canvas != 0, axis=(0, 2))
    if np.any(rows) and np.any(cols):
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        canvas = canvas[rmin:rmax+1, cmin:cmax+1]

    cv2.imwrite(filename, canvas)
    print(f"縫合完成: {filename}")

# ==========================================
# 1. SIFT 流程 (原圖大小, 全域搜尋)
# ==========================================
def run_sift_pipeline(img1, img2):
    print("\n--- 執行 SIFT (原圖大小) ---")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("SIFT 找不到特徵")
        return

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # RANSAC
    if len(pts1) < 4:
        print("SIFT 點數不足")
        return

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()
    
    pts1_in = pts1[np.array(matches_mask) == 1]
    pts2_in = pts2[np.array(matches_mask) == 1]
    
    print(f"SIFT RANSAC 後有效點數: {len(pts1_in)}")

    # 畫圖
    vis = draw_matches_with_roi(img1, img2, pts1_in, pts2_in, None, None, "SIFT", (255, 0, 0))
    cv2.imwrite("result_1_SIFT_match.jpg", vis)
    
    # 縫合
    if H is not None:
        simple_stitch(img1, img2, H, "result_1_SIFT_stitch.jpg")

# ==========================================
# 2. LoFTR 流程 (ROI + 1280x960 + Conf 0.3)
# ==========================================
def run_loftr_roi_pipeline(img1, img2):
    print("\n--- 執行 LoFTR (ROI + 1280x960 + 0.3) ---")
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 設定 ROI: img1 取下半部 50%, img2 取上半部 50%
    roi_h1 = int(h1 * 0.5)
    roi_y1 = h1 - roi_h1
    roi1 = img1[roi_y1:h1, 0:w1] # img1 的底部
    
    roi_h2 = int(h2 * 0.5)
    roi_y2 = 0
    roi2 = img2[0:roi_h2, 0:w2] # img2 的頂部

    # 準備輸入 LoFTR (縮放到 1280x960)
    infer_w, infer_h = 1280, 960
    input1 = cv2.resize(roi1, (infer_w, infer_h))
    input2 = cv2.resize(roi2, (infer_w, infer_h))
    
    img1_tensor = kornia.image_to_tensor(cv2.cvtColor(input1, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    img2_tensor = kornia.image_to_tensor(cv2.cvtColor(input2, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained='outdoor').to(device)
    
    with torch.no_grad():
        res = matcher({"image0": img1_tensor.to(device), "image1": img2_tensor.to(device)})

    mkpts0 = res['keypoints0'].cpu().numpy()
    mkpts1 = res['keypoints1'].cpu().numpy()
    conf = res['confidence'].cpu().numpy()

    # 信心度過濾 (0.3)
    valid = conf > 0.3
    pts1 = mkpts0[valid]
    pts2 = mkpts1[valid]
    
    print(f"LoFTR 原始匹配點: {len(pts1)}")

    # --- 座標還原三部曲 ---
    # 1. 從 1280x960 還原回 ROI 尺寸
    scale_x1 = w1 / infer_w
    scale_y1 = roi_h1 / infer_h
    scale_x2 = w2 / infer_w
    scale_y2 = roi_h2 / infer_h
    
    pts1[:, 0] *= scale_x1
    pts1[:, 1] *= scale_y1
    pts2[:, 0] *= scale_x2
    pts2[:, 1] *= scale_y2
    
    # 2. 從 ROI 還原回原圖座標 (加上 Offset)
    pts1[:, 1] += roi_y1 # img1 是取底部，所以 y 要加 roi_y1
    # pts2 是取頂部，y 不用加
    
    # 3. RANSAC 過濾
    if len(pts1) < 4:
        print("LoFTR 點數不足")
        return

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()
    
    pts1_in = pts1[np.array(matches_mask) == 1]
    pts2_in = pts2[np.array(matches_mask) == 1]

    print(f"LoFTR RANSAC 後有效點數: {len(pts1_in)}")

    # 畫圖 (原圖大小，畫出 ROI 框)
    # ROI 座標格式 (x, y, w, h)
    roi1_rect = (0, roi_y1, w1, roi_h1)
    roi2_rect = (0, 0, w2, roi_h2)
    
    vis = draw_matches_with_roi(img1, img2, pts1_in, pts2_in, roi1_rect, roi2_rect, "LoFTR", (0, 255, 0))
    cv2.imwrite("result_2_LoFTR_match.jpg", vis)
    
    # 縫合
    if H is not None:
        simple_stitch(img1, img2, H, "result_2_LoFTR_stitch.jpg")

if __name__ == "__main__":
    img1_path = "image1.jpg" # 上圖
    img2_path = "image2.jpg" # 下圖
    
    if os.path.exists(img1_path):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # 1. 跑 SIFT
        run_sift_pipeline(img1, img2)
        
        # 2. 跑 LoFTR
        run_loftr_roi_pipeline(img1, img2)
        
        print("\n程式執行完畢，請查看結果圖片。")
    else:
        print("找不到圖片")
