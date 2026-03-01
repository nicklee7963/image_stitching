import cv2
import torch
import kornia
from kornia.feature import LoFTR
import numpy as np
import os

# ==========================================
# 繪圖工具
# ==========================================
def draw_matches_side_by_side(img1, img2, pts1, pts2, color=(0, 255, 0), filename="strict_matches.jpg"):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    vis_h = max(h1, h2)
    vis_w = w1 + w2
    vis_img = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    
    vis_img[:h1, :w1] = img1
    vis_img[:h2, w1:w1+w2] = img2

    # 為了看清楚，隨機抽 50 條線畫
    indices = np.arange(len(pts1))
    np.random.shuffle(indices)
    draw_count = min(len(pts1), 50)
    
    for i in range(draw_count):
        idx = indices[i]
        pt1 = (int(pts1[idx][0]), int(pts1[idx][1]))
        pt2 = (int(pts2[idx][0] + w1), int(pts2[idx][1]))
        
        cv2.line(vis_img, pt1, pt2, color, 2)
        cv2.circle(vis_img, pt1, 5, (0, 0, 255), -1)
        cv2.circle(vis_img, pt2, 5, (0, 0, 255), -1)

    print(f"已儲存圖片: {filename}")
    cv2.imwrite(filename, vis_img)

# ==========================================
# 嚴格版 LoFTR 匹配流程
# ==========================================
def run_strict_loftr_matching(img1_path, img2_path):
    print("\n--- 執行 LoFTR 特徵匹配 (嚴格模式) ---")
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("圖片讀取失敗")
        return

    # 統一寬度
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if w1 != w2:
        img2 = cv2.resize(img2, (w1, int(h2 * w1 / w2)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained='outdoor').to(device)

    # 縮圖計算
    infer_w, infer_h = 640, 480
    img1_tensor = kornia.image_to_tensor(cv2.cvtColor(cv2.resize(img1, (infer_w, infer_h)), cv2.COLOR_BGR2GRAY), False).float() / 255.0
    img2_tensor = kornia.image_to_tensor(cv2.cvtColor(cv2.resize(img2, (infer_w, infer_h)), cv2.COLOR_BGR2GRAY), False).float() / 255.0
    
    with torch.no_grad():
        res = matcher({"image0": img1_tensor.to(device), "image1": img2_tensor.to(device)})

    mkpts0 = res['keypoints0'].cpu().numpy()
    mkpts1 = res['keypoints1'].cpu().numpy()
    conf = res['confidence'].cpu().numpy()

    # 1. 提高信心度門檻 (關鍵調整!)
    # 只取 LoFTR 非常非常有把握的點
    conf_threshold = 0.7  
    valid = conf > conf_threshold
    mkpts0 = mkpts0[valid]
    mkpts1 = mkpts1[valid]
    
    print(f"信心度 > {conf_threshold} 的點數: {len(mkpts0)}")

    # 2. 座標還原
    scale_x1 = img1.shape[1] / infer_w
    scale_y1 = img1.shape[0] / infer_h
    scale_x2 = img2.shape[1] / infer_w
    scale_y2 = img2.shape[0] / infer_h
    
    mkpts0[:, 0] *= scale_x1
    mkpts0[:, 1] *= scale_y1
    mkpts1[:, 0] *= scale_x2
    mkpts1[:, 1] *= scale_y2

    # 3. 嚴格的 RANSAC (關鍵調整!)
    if len(mkpts0) < 4:
        print("高信心點數不足，無法執行 RANSAC")
        # 還是畫出來看看，但用不同顏色 (紅色) 表示沒通過驗證
        draw_matches_side_by_side(img1, img2, mkpts0, mkpts1, color=(0, 0, 255), filename="strict_matches_failed.jpg")
        return

    # 使用 estimateAffinePartial2D (只允許平移+旋轉+縮放)
    # 並且把容許誤差 (ransacReprojThreshold) 設得很小 (例如 3.0)
    M, inliers = cv2.estimateAffinePartial2D(mkpts0, mkpts1, ransacReprojThreshold=3.0)
    
    if M is None:
        print("RANSAC 幾何驗證失敗，沒有找到一致的變換。")
        draw_matches_side_by_side(img1, img2, mkpts0, mkpts1, color=(0, 0, 255), filename="strict_matches_failed.jpg")
        return

    # 提取 Inliers
    inliers = inliers.ravel().astype(bool)
    pts1_in = mkpts0[inliers]
    pts2_in = mkpts1[inliers]
    
    print(f"嚴格 RANSAC 後有效點數 (Inliers): {len(pts1_in)}")
    print(f"Inlier Ratio: {len(pts1_in)/len(mkpts0):.2%}")

    # 畫圖 (綠色代表通過嚴格驗證)
    draw_matches_side_by_side(img1, img2, pts1_in, pts2_in, color=(0, 255, 0), filename="strict_matches_success.jpg")

if __name__ == "__main__":
    # 請替換成你的圖片路徑
    run_strict_loftr_matching("image1.jpg", "image2.jpg")
