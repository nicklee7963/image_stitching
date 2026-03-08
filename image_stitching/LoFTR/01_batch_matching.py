import cv2
import torch
import kornia as K
import kornia.feature as KF
import numpy as np
import os
from pathlib import Path

# --- 設定區域 ---
IMG_DIR = Path("../images/20251205")
# 在這裡加入你想要測試的所有解析度 (寬度)
RESOLUTIONS = [640, 800, 1024] 
RANSAC_THRESH = 1.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 LoFTR
matcher = KF.LoFTR(pretrained='outdoor').to(DEVICE).eval()

def run_loftr_workflow(res_width):
    """針對特定解析度執行完整流程"""
    # 建立對應解析度的儲存資料夾
    save_dir = Path(f"./Result/Original_{res_width}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 依照數字排序處理 plant_0 到 plant_20
    plant_folders = sorted(list(IMG_DIR.glob("plant_*")), key=lambda x: int(x.name.split('_')[1]))

    for plant_folder in plant_folders:
        all_files = list(plant_folder.glob("*.[jJ][pP][gG]")) + list(plant_folder.glob("*.[pP][nN][gG]"))
        img_up_path = next((f for f in all_files if "up" in f.name.lower()), None)
        img_down_path = next((f for f in all_files if "down" in f.name.lower()), None)

        if not img_up_path or not img_down_path:
            continue

        # 1. 讀取
        img_up = cv2.imread(str(img_up_path))
        img_down = cv2.imread(str(img_down_path))

        # 2. 縮放與對齊高度
        h1, w1 = img_up.shape[:2]
        ratio1 = res_width / float(w1)
        target_h = int(h1 * ratio1)
        img1_resized = cv2.resize(img_up, (res_width, target_h))

        h2, w2 = img_down.shape[:2]
        img2_resized = cv2.resize(img_down, (int(w2 * (target_h / float(h2))), target_h))

        # 3. LoFTR 匹配
        t1 = K.image_to_tensor(cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY), keepdim=False).float() / 255.
        t2 = K.image_to_tensor(cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY), keepdim=False).float() / 255.

        input_dict = {"image0": t1.to(DEVICE), "image1": t2.to(DEVICE)}
        with torch.no_grad():
            correspondences = matcher(input_dict)

        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()

        # 4. RANSAC
        if len(mkpts0) > 4:
            _, mask = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.FM_RANSAC, RANSAC_THRESH)
            mask = mask.ravel() > 0
            inliers_count = np.sum(mask)
            m0, m1 = mkpts0[mask], mkpts1[mask]
        else:
            inliers_count = 0
            m0, m1 = mkpts0, mkpts1

        # 5. 繪製
        vis_img = np.hstack((img1_resized, img2_resized))
        for p1, p2 in zip(m0, m1):
            pt1, pt2 = (int(p1[0]), int(p1[1])), (int(p2[0] + img1_resized.shape[1]), int(p2[1]))
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

        # 文字資訊
        info = [f"Matches: {inliers_count}", f"Res: {res_width}x{target_h}"]
        for i, text in enumerate(info):
            cv2.putText(vis_img, text, (20, 50 + (i * 45)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        # 6. 儲存
        save_path = save_dir / f"{plant_folder.name}_matched.jpg"
        cv2.imwrite(str(save_path), vis_img)

    print(f"--- 解析度 {res_width} 處理完成 ---")

if __name__ == "__main__":
    for res in RESOLUTIONS:
        print(f"開始執行解析度: {res}")
        run_loftr_workflow(res)
    print("\n所有任務已完成！")
