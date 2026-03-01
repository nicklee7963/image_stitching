import cv2
import torch
import kornia
from kornia.feature import LoFTR
import numpy as np
import os

# ==========================================
# 繪圖工具
# ==========================================
def draw_matches_side_by_side(img1, img2, pts1, pts2, color=(0, 255, 0), filename="cropped_matches.jpg"):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    vis_h = max(h1, h2)
    vis_w = w1 + w2
    vis_img = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    
    vis_img[:h1, :w1] = img1
    vis_img[:h2, w1:w1+w2] = img2

    # 隨機抽 50 條線畫
    if len(pts1) > 50:
        indices = np.random.choice(len(pts1), 50, replace=False)
    else:
        indices = range(len(pts1))

    for i in indices:
        pt1 = (int(pts1[i][0]), int(pts1[i][1]))
        pt2 = (int(pts2[i][0] + w1), int(pts2[i][1]))
        cv2.line(vis_img, pt1, pt2, color, 2)
        cv2.circle(vis_img, pt1, 5, (0, 0, 255), -1)
        cv2.circle(vis_img, pt2, 5, (0, 0, 255), -1)

    print(f"已儲存圖片: {filename}")
    cv2.imwrite(filename, vis_img)

# ==========================================
# 裁切與匹配邏輯
# ==========================================
def run_overlap_crop_test(img1_path, img2_path, keep_ratio=0.5):
    print(f"--- 執行裁切匹配測試 (保留比例: {keep_ratio}) ---")
    
    # 假設 img1 是上面那張，img2 是下面那張
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("圖片讀取失敗")
        return

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # --- 1. 執行裁切 ---
    # img1 (上圖): 我們要它的「下半部」 -> 切掉上面 (1-ratio)
    crop_h1 = int(h1 * keep_ratio)
    start_y1 = h1 - crop_h1
    img1_crop = img1[start_y1:, :] # 取底部
    
    # img2 (下圖): 我們要它的「上半部」 -> 切掉下面 (1-ratio)
    crop_h2 = int(h2 * keep_ratio)
    img2_crop = img2[:crop_h2, :] # 取頂部

    print(f"圖1 (上) 保留區域: {w1}x{crop_h1} (底部)")
    print(f"圖2 (下) 保留區域: {w2}x{crop_h2} (頂部)")

    # 存下來確認一下切得對不對
    cv2.imwrite("debug_crop1.jpg", img1_crop)
    cv2.imwrite("debug_crop2.jpg", img2_crop)

    # --- 2. 執行 LoFTR ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained='outdoor').to(device)

    # 縮圖計算
    infer_w, infer_h = 640, 480
    input1 = cv2.resize(img1_crop, (infer_w, infer_h))
    input2 = cv2.resize(img2_crop, (infer_w, infer_h))
    
    img1_tensor = kornia.image_to_tensor(cv2.cvtColor(input1, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    img2_tensor = kornia.image_to_tensor(cv2.cvtColor(input2, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    
    with torch.no_grad():
        res = matcher({"image0": img1_tensor.to(device), "image1": img2_tensor.to(device)})

    mkpts0 = res['keypoints0'].cpu().numpy()
    mkpts1 = res['keypoints1'].cpu().numpy()
    conf = res['confidence'].cpu().numpy()

    # 過濾
    valid = conf > 0.6 # 信心度設高一點
    mkpts0 = mkpts0[valid]
    mkpts1 = mkpts1[valid]
    
    # 座標還原回 "裁切後" 的尺寸
    scale_x1 = w1 / infer_w
    scale_y1 = crop_h1 / infer_h
    scale_x2 = w2 / infer_w
    scale_y2 = crop_h2 / infer_h
    
    mkpts0[:, 0] *= scale_x1
    mkpts0[:, 1] *= scale_y1
    mkpts1[:, 0] *= scale_x2
    mkpts1[:, 1] *= scale_y2

    print(f"找到高信心點數: {len(mkpts0)}")

    if len(mkpts0) < 4:
        print("點數不足，無法執行 RANSAC")
        return

    # --- 3. RANSAC 驗證 ---
    # 這裡我們用 Affine，因為這兩塊理論上是緊密連接的
    M, inliers = cv2.estimateAffinePartial2D(mkpts0, mkpts1)
    
    if M is None:
        print("RANSAC 失敗，這兩塊區域可能真的沒重疊。")
        draw_matches_side_by_side(img1_crop, img2_crop, mkpts0, mkpts1, (0,0,255), "cropped_failed.jpg")
    else:
        inliers = inliers.ravel().astype(bool)
        pts1_in = mkpts0[inliers]
        pts2_in = mkpts1[inliers]
        print(f"有效匹配點 (Inliers): {len(pts1_in)}")
        
        # 畫出結果 (綠色)
        draw_matches_side_by_side(img1_crop, img2_crop, pts1_in, pts2_in, (0, 255, 0), "cropped_success.jpg")

if __name__ == "__main__":
    # image1 放「上半部」的照片
    # image2 放「下半部」的照片
    run_overlap_crop_test("image1.jpg", "image2.jpg", keep_ratio=0.5)
