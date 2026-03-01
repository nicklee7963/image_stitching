import cv2
import torch
import kornia
from kornia.feature import LoFTR
import numpy as np
import os

def ensure_same_size(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1 != h2) or (w1 != w2):
        print(f"調整圖片尺寸: {w2}x{h2} -> {w1}x{h1}")
        img2 = cv2.resize(img2, (w1, h1))
    return img1, img2

# ==========================================
# [新版] 穩定型縫合 (使用 Affine 仿射變換)
# ==========================================
def stitch_images_affine(img1, img2, pts1, pts2, method_name):
    print(f"--- [{method_name}] 使用 Affine 模式計算縫合... ---")
    
    if len(pts1) < 3: # Affine 只需要 3 點
        print("點數不足，無法計算。")
        return None

    # 改用 estimateAffinePartial2D
    # 這會限制變換只能是：平移 + 旋轉 + 縮放 (不會有奇怪的透視扭曲)
    # 這是解決「全黑」或「過度拉伸」的最佳解法
    M, inliers = cv2.estimateAffinePartial2D(pts2, pts1)

    if M is None:
        print("無法計算 Affine 矩陣")
        return None

    # M 是一個 2x3 矩陣
    print(f"[{method_name}] 計算出的變換矩陣:\n{M}")

    # --- 計算畫布範圍 ---
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 取得 img2 的四個角
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    # 使用 transform (注意 Affine 用 transform 不是 perspectiveTransform)
    corners_img2_trans = cv2.transform(corners_img2, M)
    
    # 取得 img1 的四個角
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    # 合併邊界
    all_corners = np.concatenate((corners_img1, corners_img2_trans), axis=0)
    
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # 計算平移修正 (如果跑出負座標，要移回來)
    tx, ty = -xmin, -ymin
    
    # 建立新的 2x3 變換矩陣 (原矩陣 + 平移)
    M_final = M.copy()
    M_final[0, 2] += tx
    M_final[1, 2] += ty

    # 建立畫布
    canvas_w = xmax - xmin
    canvas_h = ymax - ymin
    print(f"預計畫布: {canvas_w}x{canvas_h}")

    # 執行 WarpAffine
    warped_img2 = cv2.warpAffine(img2, M_final, (canvas_w, canvas_h))

    # 貼上左圖
    stitched_result = warped_img2.copy()
    
    # 左圖的新位置
    x_start, y_start = tx, ty
    
    try:
        # 簡單覆蓋
        stitched_result[y_start:y_start+h1, x_start:x_start+w1] = img1
    except:
        pass

    return stitched_result

# ==========================================
# 2. LoFTR 流程
# ==========================================
def run_loftr_pipeline(img1, img2):
    print("\n=== 執行 LoFTR (穩定版) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained='outdoor').to(device)

    h_orig, w_orig = img1.shape[:2]
    TARGET_W, TARGET_H = 640, 480 
    
    img1_resized = cv2.resize(img1, (TARGET_W, TARGET_H))
    img2_resized = cv2.resize(img2, (TARGET_W, TARGET_H))

    img1_tensor = kornia.image_to_tensor(cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    img2_tensor = kornia.image_to_tensor(cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    
    with torch.no_grad():
        res = matcher({"image0": img1_tensor.to(device), "image1": img2_tensor.to(device)})

    mkpts0 = res['keypoints0'].cpu().numpy()
    mkpts1 = res['keypoints1'].cpu().numpy()
    conf = res['confidence'].cpu().numpy()

    # 嚴格一點，只取最有把握的點，避免被雜訊干擾
    valid = conf > 0.5
    mkpts0 = mkpts0[valid]
    mkpts1 = mkpts1[valid]
    
    # 座標還原
    scale_x = w_orig / TARGET_W
    scale_y = h_orig / TARGET_H
    mkpts0[:, 0] *= scale_x
    mkpts0[:, 1] *= scale_y
    mkpts1[:, 0] *= scale_x
    mkpts1[:, 1] *= scale_y

    print(f"高信心度點數: {len(mkpts0)}")

    # 畫連線圖 (確認用)
    vis_img = np.zeros((h_orig, w_orig * 2, 3), dtype=np.uint8)
    vis_img[:h_orig, :w_orig] = img1
    vis_img[:h_orig, w_orig:w_orig*2] = img2
    limit = min(len(mkpts0), 50)
    for i in range(limit):
        pt1 = (int(mkpts0[i][0]), int(mkpts0[i][1]))
        pt2 = (int(mkpts1[i][0] + w_orig), int(mkpts1[i][1]))
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)
        cv2.circle(vis_img, pt1, 5, (0, 0, 255), -1)
        cv2.circle(vis_img, pt2, 5, (0, 0, 255), -1)

    # 執行穩定縫合
    stitch_img = stitch_images_affine(img1, img2, mkpts0, mkpts1, "LoFTR")
    
    return vis_img, stitch_img

if __name__ == "__main__":
    image_path_1 = "image1.jpg" 
    image_path_2 = "image2.jpg"

    if os.path.exists(image_path_1) and os.path.exists(image_path_2):
        img1 = cv2.imread(image_path_1)
        img2 = cv2.imread(image_path_2)
        img1, img2 = ensure_same_size(img1, img2)
        
        vis, result = run_loftr_pipeline(img1, img2)
        
        if vis is not None:
            cv2.imwrite("result_lines.jpg", vis)
        if result is not None:
            cv2.imwrite("result_affine_stitch.jpg", result)
            print(">>> 成功！請查看 result_affine_stitch.jpg")
    else:
        print("找無圖片")
