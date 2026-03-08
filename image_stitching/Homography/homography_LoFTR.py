import cv2
import torch
import numpy as np
import os
import glob
import kornia as K
from kornia.feature import LoFTR

# 設定路徑
BASE_IMAGE_DIR = "../images/20251205"
RESULT_DIR = "Result_LoFTR"

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def match_loftr(img1, img2, device):
    """使用 LoFTR 提取特徵點"""
    # 為了避免 VRAM 爆掉，縮小解析度進行匹配 (可依據你的顯存微調 MAX_SIZE)
    MAX_SIZE = 840 
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    scale1 = MAX_SIZE / max(h1, w1)
    scale2 = MAX_SIZE / max(h2, w2)
    
    img1_resized = cv2.resize(img1, (int(w1 * scale1), int(h1 * scale1)))
    img2_resized = cv2.resize(img2, (int(w2 * scale2), int(h2 * scale2)))
    
    gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    
    # 轉為 PyTorch Tensor 並正規化到 0~1
    tensor1 = K.image_to_tensor(gray1, False).float() / 255.
    tensor2 = K.image_to_tensor(gray2, False).float() / 255.
    
    tensor1 = tensor1.to(device)
    tensor2 = tensor2.to(device)
    
    # 載入預訓練模型 (戶外場景)
    matcher = LoFTR(pretrained='outdoor').to(device)
    
    input_dict = {"image0": tensor1, "image1": tensor2}
    with torch.inference_mode():
        correspondences = matcher(input_dict)
        
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    
    # 將找到的座標放大回原始解析度的位置
    mkpts0 = mkpts0 / scale1
    mkpts1 = mkpts1 / scale2
    
    return mkpts0, mkpts1

def stitch_loftr_homography(up_path, down_path, plant_name, device):
    img_up = cv2.imread(up_path)
    img_down = cv2.imread(down_path)
    
    if img_up is None or img_down is None:
        return
        
    print(f"[{plant_name}] 正在使用 LoFTR 擷取特徵...")
    mkpts_up, mkpts_down = match_loftr(img_up, img_down, device)
    
    if len(mkpts_up) < 4:
        print(f"[{plant_name}] LoFTR 匹配點不足。")
        return

    print(f"[{plant_name}] 找到 {len(mkpts_up)} 個特徵對，計算單應性矩陣...")
    # 使用 LoFTR 找出的高信賴度點來算 H 矩陣
    H, mask = cv2.findHomography(mkpts_up, mkpts_down, cv2.RANSAC, 5.0)
    
    if H is None:
        print(f"[{plant_name}] H 矩陣計算失敗。")
        return

    h_up, w_up = img_up.shape[:2]
    h_down, w_down = img_down.shape[:2]

    # 計算 Up.jpg 變形後的位置
    corners_up = np.float32([[0, 0], [0, h_up], [w_up, h_up], [w_up, 0]]).reshape(-1, 1, 2)
    warped_corners_up = cv2.perspectiveTransform(corners_up, H)
    
    # 加入 Down.jpg 的位置以決定總畫布大小
    corners_down = np.float32([[0, 0], [0, h_down], [w_down, h_down], [w_down, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((warped_corners_up, corners_down), axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    
    # 安全機制防當機
    if canvas_w > 15000 or canvas_h > 15000:
        print(f"[{plant_name}] 矩陣算出畫布過大 ({canvas_w}x{canvas_h})，跳過防當機。")
        return

    # 執行 Up.jpg 的變形
    canvas = cv2.warpPerspective(img_up, H_translation.dot(H), (canvas_w, canvas_h))
    
    # 疊加 Down.jpg
    down_y_start = translation_dist[1]
    down_x_start = translation_dist[0]
    canvas[down_y_start:down_y_start+h_down, down_x_start:down_x_start+w_down] = img_down

    result_path = os.path.join(RESULT_DIR, f"{plant_name}_loftr_stitch.jpg")
    cv2.imwrite(result_path, canvas)
    print(f"[{plant_name}] 縫合完成，儲存至 {result_path}\n")

def main():
    # 自動偵測是否可用 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用裝置: {device}\n")
    
    plant_folders = glob.glob(os.path.join(BASE_IMAGE_DIR, "plant_*"))
    for folder in plant_folders:
        plant_name = os.path.basename(folder)
        stitch_loftr_homography(os.path.join(folder, "Up.jpg"), os.path.join(folder, "Down.jpg"), plant_name, device)

if __name__ == "__main__":
    main()
