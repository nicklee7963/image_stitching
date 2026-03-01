import cv2
import torch
import kornia
from kornia.feature import LoFTR
import numpy as np
import os

# ==========================================
# 1. 基礎工具
# ==========================================
def ensure_same_size(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1 != h2) or (w1 != w2):
        print(f"調整圖片尺寸: {w2}x{h2} -> {w1}x{h1}")
        img2 = cv2.resize(img2, (w1, h1))
    return img1, img2

def draw_matches_custom(img1, img2, pts1, pts2, color=(0, 255, 0)):
    h, w = img1.shape[:2]
    vis_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
    vis_img[:h, :w] = img1
    vis_img[:h, w:w*2] = img2

    # 最多畫 100 條線，避免太亂
    limit = min(len(pts1), 100)
    for i in range(limit):
        pt1 = (int(pts1[i][0]), int(pts1[i][1]))
        pt2 = (int(pts2[i][0] + w), int(pts2[i][1]))
        cv2.line(vis_img, pt1, pt2, color, 2)
        cv2.circle(vis_img, pt1, 5, (0, 0, 255), -1)
        cv2.circle(vis_img, pt2, 5, (0, 0, 255), -1)
    return vis_img

# ==========================================
# 2. 智慧型縫合函式 (防止黑屏與當機)
# ==========================================
def stitch_images_robust(img1, img2, pts1, pts2, method_name):
    print(f"--- [{method_name}] 計算縫合幾何中... ---")
    
    if len(pts1) < 4:
        print(f"[{method_name}] 錯誤：特徵點不足 4 個，無法計算矩陣。")
        return None

    # 計算 H 矩陣 (RANSAC 門檻設為 5.0，因為傳進來的點已經篩選過)
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    
    if H is None:
        print(f"[{method_name}] 錯誤：無法計算 Homography 矩陣。")
        return None

    # 計算變形後的邊界
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    corners_img2_trans = cv2.perspectiveTransform(corners_img2, H)
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    all_corners = np.concatenate((corners_img1, corners_img2_trans), axis=0)
    
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # 畫布尺寸檢查
    canvas_w = xmax - xmin
    canvas_h = ymax - ymin
    
    print(f"[{method_name}] 預計畫布尺寸: {canvas_w} x {canvas_h}")
    
    if canvas_w > 30000 or canvas_h > 30000:
        print(f"[{method_name}] 警告：畫布過大，放棄縫合。")
        return None

    # 平移矩陣
    translation_dist = [-xmin, -ymin]
    T = np.array([[1, 0, translation_dist[0]], 
                  [0, 1, translation_dist[1]], 
                  [0, 0, 1]])

    final_transform = T.dot(H)
    
    # 執行變形
    warped_img2 = cv2.warpPerspective(img2, final_transform, (canvas_w, canvas_h))
    
    # 貼上左圖
    stitched_result = warped_img2.copy()
    x_start, y_start = translation_dist[0], translation_dist[1]
    
    try:
        stitched_result[y_start:y_start+h1, x_start:x_start+w1] = img1
    except:
        pass # 忽略邊界誤差

    return stitched_result

# ==========================================
# 3. SIFT 流程 (傳統演算法)
# ==========================================
def run_sift_pipeline(img1, img2):
    print("\n=== 正在執行 SIFT 流程 ===")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 特徵匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    print(f"[SIFT] 原始匹配點數: {len(pts1)}")

    # RANSAC 過濾
    if len(pts1) < 4:
        return None, None
    
    # SIFT 點通常較多但雜訊也多，這裡用標準 RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()
    
    pts1_in = pts1[np.array(matches_mask) == 1]
    pts2_in = pts2[np.array(matches_mask) == 1]
    
    print(f"[SIFT] RANSAC 後有效點數: {len(pts1_in)}")
    print(f"[SIFT] Inlier Ratio: {len(pts1_in)/len(pts1):.2%}")

    # 產生連線圖
    vis_img = draw_matches_custom(img1, img2, pts1_in, pts2_in, color=(0, 255, 255)) # 黃色線

    # 產生縫合圖
    stitch_img = stitch_images_robust(img1, img2, pts1_in, pts2_in, "SIFT")
    
    return vis_img, stitch_img

# ==========================================
# 4. LoFTR 流程 (深度學習)
# ==========================================
def run_loftr_pipeline(img1, img2):
    print("\n=== 正在執行 LoFTR 流程 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained='outdoor').to(device)

    h_orig, w_orig = img1.shape[:2]
    TARGET_W, TARGET_H = 640, 480 # 縮圖計算
    
    img1_resized = cv2.resize(img1, (TARGET_W, TARGET_H))
    img2_resized = cv2.resize(img2, (TARGET_W, TARGET_H))

    img1_tensor = kornia.image_to_tensor(cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    img2_tensor = kornia.image_to_tensor(cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    
    with torch.no_grad():
        res = matcher({"image0": img1_tensor.to(device), "image1": img2_tensor.to(device)})

    mkpts0 = res['keypoints0'].cpu().numpy()
    mkpts1 = res['keypoints1'].cpu().numpy()
    conf = res['confidence'].cpu().numpy()

    # 參數設定
    conf_threshold = 0.2
    valid = conf > conf_threshold
    mkpts0 = mkpts0[valid]
    mkpts1 = mkpts1[valid]
    
    # 座標還原
    scale_x = w_orig / TARGET_W
    scale_y = h_orig / TARGET_H
    mkpts0[:, 0] *= scale_x
    mkpts0[:, 1] *= scale_y
    mkpts1[:, 0] *= scale_x
    mkpts1[:, 1] *= scale_y

    print(f"[LoFTR] 原始匹配點數: {len(mkpts0)}")

    # RANSAC 過濾 (針對大圖放寬容忍度)
    if len(mkpts0) < 4:
        return None, None

    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 20.0)
    matches_mask = mask.ravel().tolist()
    
    pts1_in = mkpts0[np.array(matches_mask) == 1]
    pts2_in = mkpts1[np.array(matches_mask) == 1]
    
    print(f"[LoFTR] RANSAC 後有效點數: {len(pts1_in)}")
    print(f"[LoFTR] Inlier Ratio: {len(pts1_in)/len(mkpts0):.2%}")

    # 產生連線圖
    vis_img = draw_matches_custom(img1, img2, pts1_in, pts2_in, color=(0, 255, 0)) # 綠色線

    # 產生縫合圖
    stitch_img = stitch_images_robust(img1, img2, pts1_in, pts2_in, "LoFTR")
    
    return vis_img, stitch_img

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    image_path_1 = "image1.jpg" 
    image_path_2 = "image2.jpg"

    if not os.path.exists(image_path_1) or not os.path.exists(image_path_2):
        print("錯誤：找不到圖片")
    else:
        try:
            print("1. 讀取圖片中...")
            img1_raw = cv2.imread(image_path_1)
            img2_raw = cv2.imread(image_path_2)
            img1, img2 = ensure_same_size(img1_raw, img2_raw)

            # --- 執行 SIFT ---
            sift_vis, sift_stitch = run_sift_pipeline(img1, img2)
            if sift_vis is not None:
                cv2.imwrite("result_SIFT_matches.jpg", sift_vis)
            if sift_stitch is not None:
                cv2.imwrite("result_SIFT_stitched.jpg", sift_stitch)
                print(">> SIFT 結果已存檔")

            # --- 執行 LoFTR ---
            loftr_vis, loftr_stitch = run_loftr_pipeline(img1, img2)
            if loftr_vis is not None:
                cv2.imwrite("result_LoFTR_matches.jpg", loftr_vis)
            if loftr_stitch is not None:
                cv2.imwrite("result_LoFTR_stitched.jpg", loftr_stitch)
                print(">> LoFTR 結果已存檔")
                
            print("\n全部完成！請查看當前資料夾下的 4 張圖片。")

        except Exception as e:
            print(f"程式執行發生錯誤: {e}")
            import traceback
            traceback.print_exc()
