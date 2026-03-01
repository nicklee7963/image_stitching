import cv2
import torch
import kornia
from kornia.feature import LoFTR
import matplotlib.pyplot as plt
import numpy as np

def show_image(title, img):
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# 統一尺寸工具
def ensure_same_size(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1 != h2) or (w1 != w2):
        img2 = cv2.resize(img2, (w1, h1))
    return img1, img2

# ==========================================
# [核心功能] RANSAC 過濾器
# ==========================================
def apply_ransac(pts1, pts2, threshold=5.0):
    """
    輸入兩組點 (N, 2)，回傳經過 RANSAC 過濾後的 Inliers
    threshold: 容許的誤差範圍 (像素)，設 3.0~5.0 通常適合
    """
    if len(pts1) < 4:
        print("點數過少 (<4)，無法執行 RANSAC")
        return pts1, pts2, 0.0

    # 使用 findHomography 計算單應性矩陣，並找出 Inliers
    # cv2.RANSAC: 使用 RANSAC 演算法
    # mask: 1 代表 Inlier, 0 代表 Outlier
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold)
    
    if mask is None:
        return np.array([]), np.array([]), 0.0

    matches_mask = mask.ravel().tolist()
    
    # 篩選出 Inliers
    pts1_inliers = pts1[np.array(matches_mask) == 1]
    pts2_inliers = pts2[np.array(matches_mask) == 1]
    
    inlier_ratio = len(pts1_inliers) / len(pts1)
    return pts1_inliers, pts2_inliers, inlier_ratio

# ==========================================
# 1. SIFT + RANSAC
# ==========================================
def run_sift_ransac(img1, img2):
    print(f"\n--- [SIFT] 執行中 ---")
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 提取座標點
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    print(f"原始匹配數量 (Before RANSAC): {len(pts1)}")

    # [執行 RANSAC]
    pts1_in, pts2_in, ratio = apply_ransac(pts1, pts2)

    print(f"過濾後數量 (After RANSAC): {len(pts1_in)}")
    print(f"Inlier Ratio (準確率): {ratio:.2%}") # 這是論文重點數據

    # 繪圖
    h, w = img1.shape[:2]
    vis_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
    vis_img[:h, :w] = img1
    vis_img[:h, w:w*2] = img2

    for i in range(len(pts1_in)):
        pt1 = (int(pts1_in[i][0]), int(pts1_in[i][1]))
        pt2 = (int(pts2_in[i][0] + w), int(pts2_in[i][1]))
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2) # 綠色線
        cv2.circle(vis_img, pt1, 5, (0, 0, 255), -1)
        cv2.circle(vis_img, pt2, 5, (0, 0, 255), -1)

    return vis_img

# ==========================================
# 2. LoFTR + RANSAC
# ==========================================
def run_loftr_ransac(img1, img2):
    print(f"\n--- [LoFTR] 執行中 ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained='outdoor').to(device)

    h_orig, w_orig = img1.shape[:2]
    TARGET_W, TARGET_H = 640, 480
    
    img1_resized = cv2.resize(img1, (TARGET_W, TARGET_H))
    img2_resized = cv2.resize(img2, (TARGET_W, TARGET_H))

    img1_tensor = kornia.image_to_tensor(cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    img2_tensor = kornia.image_to_tensor(cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    
    input_dict = {"image0": img1_tensor.to(device), "image1": img2_tensor.to(device)}

    with torch.no_grad():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    confidence = correspondences['confidence'].cpu().numpy()

    # 基本信心度過濾
    conf_threshold = 0.5
    valid = confidence > conf_threshold
    mkpts0 = mkpts0[valid]
    mkpts1 = mkpts1[valid]
    
    # 座標還原 (還原到原始大圖尺寸)
    scale_x = w_orig / TARGET_W
    scale_y = h_orig / TARGET_H
    
    mkpts0[:, 0] *= scale_x
    mkpts0[:, 1] *= scale_y
    mkpts1[:, 0] *= scale_x
    mkpts1[:, 1] *= scale_y

    print(f"原始匹配數量 (Before RANSAC): {len(mkpts0)}")

    # [執行 RANSAC]
    # 注意：因為座標已經還原回大圖，這裡的 threshold=5.0 對應的是原始圖片的像素
    pts1_in, pts2_in, ratio = apply_ransac(mkpts0, mkpts1, threshold=10.0) # 針對大圖稍微放寬容忍度

    print(f"過濾後數量 (After RANSAC): {len(pts1_in)}")
    print(f"Inlier Ratio (準確率): {ratio:.2%}") # 這是論文重點數據

    # 繪圖
    vis_img = np.zeros((h_orig, w_orig * 2, 3), dtype=np.uint8)
    vis_img[:h_orig, :w_orig] = img1
    vis_img[:h_orig, w_orig:w_orig*2] = img2

    for i in range(len(pts1_in)):
        pt1 = (int(pts1_in[i][0]), int(pts1_in[i][1]))
        pt2 = (int(pts2_in[i][0] + w_orig), int(pts2_in[i][1]))
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)
        cv2.circle(vis_img, pt1, 8, (0, 0, 255), -1)
        cv2.circle(vis_img, pt2, 8, (0, 0, 255), -1)

    return vis_img

if __name__ == "__main__":
    image_path_1 = "image1.jpg" 
    image_path_2 = "image2.jpg"

    try:
        img1_raw = cv2.imread(image_path_1)
        img2_raw = cv2.imread(image_path_2)
        
        if img1_raw is None or img2_raw is None:
            raise FileNotFoundError("圖片讀取失敗")
            
        # 1. 統一尺寸
        img1_clean, img2_clean = ensure_same_size(img1_raw, img2_raw)

        # 2. 執行 SIFT + RANSAC
        sift_result = run_sift_ransac(img1_clean, img2_clean)
        show_image("SIFT (After RANSAC)", sift_result)

        # 3. 執行 LoFTR + RANSAC
        loftr_result = run_loftr_ransac(img1_clean, img2_clean)
        show_image("LoFTR (After RANSAC)", loftr_result)
        
    except Exception as e:
        print(f"錯誤: {e}")
