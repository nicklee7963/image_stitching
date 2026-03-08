import cv2
import torch
import kornia as K
import kornia.feature as KF
import numpy as np
import os
from pathlib import Path

# --- 設定 ---
IMG_DIR = Path("../images/20251205")
RESIZE_WIDTH = 800 
SAVE_DIR = Path("./Result/Stitched_LoFTR")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

matcher = KF.LoFTR(pretrained='outdoor').to(DEVICE).eval()

def vertical_stitch_loftr(plant_folder):
    all_files = list(plant_folder.glob("*.[jJ][pP][gG]")) + list(plant_folder.glob("*.[pP][nN][gG]"))
    img_up_path = next((f for f in all_files if "up" in f.name.lower()), None)
    img_down_path = next((f for f in all_files if "down" in f.name.lower()), None)

    if not img_up_path or not img_down_path: return

    # 1. 讀取與縮放
    img_u = cv2.imread(str(img_up_path))
    img_d = cv2.imread(str(img_down_path))
    
    # 統一寬度以計算 H
    def get_resized(img, w):
        h_orig, w_orig = img.shape[:2]
        return cv2.resize(img, (w, int(h_orig * (w / w_orig))))

    img1 = get_resized(img_u, RESIZE_WIDTH) # Up
    img2 = get_resized(img_d, RESIZE_WIDTH) # Down

    # 2. LoFTR 匹配
    t1 = K.image_to_tensor(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), keepdim=False).float() / 255.
    t2 = K.image_to_tensor(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), keepdim=False).float() / 255.
    
    with torch.no_grad():
        corr = matcher({"image0": t1.to(DEVICE), "image1": t2.to(DEVICE)})
    
    mkpts0, mkpts1 = corr['keypoints0'].cpu().numpy(), corr['keypoints1'].cpu().numpy()

    # 3. 計算單應性矩陣
    H, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
    if H is None: return

    # 4. 垂直縫合邏輯：計算畫布偏移
    # 我們要讓 Up 圖片投影到 Down 的坐標系，但因為 Up 在上面，y 會是負值
    # 我們需要平移矩陣 T 讓 y 軸回到正值
    h_u, w_u = img1.shape[:2]
    h_d, w_d = img2.shape[:2]
    
    # 取得 Up 圖片四個角在 Down 坐標系的位置
    corners_u = np.float32([[0, 0], [0, h_u], [w_u, h_u], [w_u, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_u, H)
    
    # 合併 Down 圖片的角來找出整體的範圍
    corners_d = np.float32([[0, 0], [0, h_d], [w_d, h_d], [w_d, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((warped_corners, corners_d), axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # 建立平移矩陣，把 y_min 移到 0
    translation_dist = [-x_min, -y_min]
    T = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # 5. 投影與拼接
    output_img = cv2.warpPerspective(img1, T.dot(H), (x_max - x_min, y_max - y_min))
    # 將 Down 圖片貼到對應位置
    output_img[translation_dist[1]:h_d + translation_dist[1], 
               translation_dist[0]:w_d + translation_dist[0]] = img2

    # 儲存到 Result
    cv2.imwrite(str(SAVE_DIR / f"{plant_folder.name}_loftr_vertical.jpg"), output_img)
    print(f"LoFTR 垂直縫合成功: {plant_folder.name}")

if __name__ == "__main__":
    plant_folders = sorted(list(IMG_DIR.glob("plant_*")), key=lambda x: int(x.name.split('_')[1]))
    for f in plant_folders: vertical_stitch_loftr(f)
