import cv2
import torch
import kornia
from kornia.feature import LoFTR
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 繪圖工具 (左右並排比較)
# ==========================================

# ==========================================
# 修正版繪圖工具 (自動統一寬度，解決 ValueError)
# ==========================================
def draw_comparison(img1, img2, pts1_sift, pts2_sift, pts1_loftr, pts2_loftr, filename="comparison_vs.jpg"):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 1. 找出最大寬度
    target_w = max(w1, w2)
    
    # 2. 補黑邊 (Padding) 讓兩張圖寬度一致
    # 如果 img1 比較窄，補右邊
    if w1 < target_w:
        pad_w = target_w - w1
        img1 = cv2.copyMakeBorder(img1, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    # 如果 img2 比較窄，補右邊
    if w2 < target_w:
        pad_w = target_w - w2
        img2 = cv2.copyMakeBorder(img2, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))

    # 更新高度 (雖然高度沒變，但為了變數清晰)
    h1 = img1.shape[0]

    # 3. 建立畫布
    # 左半邊 (SIFT)
    canvas_sift = np.vstack((img1, img2))
    # 右半邊 (LoFTR)
    canvas_loftr = np.vstack((img1, img2))
    
    # 合併左右
    final_canvas = np.hstack((canvas_sift, canvas_loftr))
    
    # 4. 畫 SIFT 連線 (左半邊)
    for p1, p2 in zip(pts1_sift, pts2_sift):
        pt1 = (int(p1[0]), int(p1[1]))
        # 下圖的 y 座標要加上上圖的高度 (h1)
        pt2 = (int(p2[0]), int(p2[1] + h1)) 
        
        cv2.line(final_canvas, pt1, pt2, (0, 0, 255), 1)
        cv2.circle(final_canvas, pt1, 3, (0, 0, 255), -1)

    # 5. 畫 LoFTR 連線 (右半邊)
    # x 座標全部要加上 target_w (因為在右邊)
    for p1, p2 in zip(pts1_loftr, pts2_loftr):
        pt1 = (int(p1[0] + target_w), int(p1[1]))
        pt2 = (int(p2[0] + target_w), int(p2[1] + h1))
        
        cv2.line(final_canvas, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(final_canvas, pt1, 3, (0, 255, 0), -1)

    # 加上文字標籤
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 調整文字位置，避免跑出畫面
    cv2.putText(final_canvas, f"SIFT: {len(pts1_sift)}", (50, 100), font, 2.0, (0, 0, 255), 4)
    cv2.putText(final_canvas, f"LoFTR: {len(pts1_loftr)}", (target_w + 50, 100), font, 2.0, (0, 255, 0), 4)

    cv2.imwrite(filename, final_canvas)
    print(f">>> 成功！比較圖已儲存: {filename}")

# ==========================================
# 1. SIFT 演算法
# ==========================================
def run_sift(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return np.array([]), np.array([])

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    
    # RANSAC 過濾雜訊
    if len(pts1) < 4: return np.array([]), np.array([])
    
    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if mask is None: return np.array([]), np.array([])
    
    matches_mask = mask.ravel().tolist()
    pts1 = pts1[np.array(matches_mask) == 1]
    pts2 = pts2[np.array(matches_mask) == 1]
    
    return pts1, pts2

# ==========================================
# 2. LoFTR 演算法
# ==========================================
def run_loftr(img1, img2):
    # 縮圖加速 (標準流程)
    infer_w, infer_h = 1280, 960
    h, w = img1.shape[:2]
    
    input1 = cv2.resize(img1, (infer_w, infer_h))
    input2 = cv2.resize(img2, (infer_w, infer_h))
    
    img1_tensor = kornia.image_to_tensor(cv2.cvtColor(input1, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    img2_tensor = kornia.image_to_tensor(cv2.cvtColor(input2, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained='outdoor').to(device)
    
    with torch.no_grad():
        res = matcher({"image0": img1_tensor.to(device), "image1": img2_tensor.to(device)})
        
    mkpts0 = res['keypoints0'].cpu().numpy()
    mkpts1 = res['keypoints1'].cpu().numpy()
    conf = res['confidence'].cpu().numpy()
    
    # 篩選 (LoFTR 的優勢在於就算把門檻調高，點還是很多)
    valid = conf > 0.3
    pts1 = mkpts0[valid]
    pts2 = mkpts1[valid]
    
    # 還原座標
    scale_x = w / infer_w
    scale_y = h / infer_h
    pts1[:, 0] *= scale_x
    pts1[:, 1] *= scale_y
    pts2[:, 0] *= scale_x
    pts2[:, 1] *= scale_y
    
    # RANSAC 過濾 (為了公平比較，也做幾何驗證)
    if len(pts1) < 4: return np.array([]), np.array([])
    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if mask is None: return np.array([]), np.array([])
    
    matches_mask = mask.ravel().tolist()
    pts1 = pts1[np.array(matches_mask) == 1]
    pts2 = pts2[np.array(matches_mask) == 1]
    
    return pts1, pts2

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    # 使用你之前裁切過、比較有重疊的圖 (效果最好)
    # 或者是那張高難度的對角線圖
    path1 = "image1.jpg" # 建議用裁切過的 ROI 圖
    path2 = "image2.jpg"
    
    # 如果你想現場裁切 (模擬 ROI)，把下面這行設為 True
    auto_crop_roi = True
    
    if os.path.exists(path1):
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        
        # 自動 ROI 裁切 (模擬你的垂直拍攝)
        if auto_crop_roi:
            h, w = img1.shape[:2]
            # 圖1留下面 50%
            img1 = img1[int(h*0.5):, :]
            # 圖2留上面 50%
            img2 = img2[:int(h*0.5), :]
        
        # 1. 跑 SIFT
        print("正在計算 SIFT...")
        p1_sift, p2_sift = run_sift(img1, img2)
        print(f"SIFT 找到有效點數: {len(p1_sift)}")
        
        # 2. 跑 LoFTR
        print("正在計算 LoFTR...")
        p1_loftr, p2_loftr = run_loftr(img1, img2)
        print(f"LoFTR 找到有效點數: {len(p1_loftr)}")
        
        # 3. 畫圖比較
        draw_comparison(img1, img2, p1_sift, p2_sift, p1_loftr, p2_loftr)
        
    else:
        print("找不到圖片")
