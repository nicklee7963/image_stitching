import cv2
import torch
import numpy as np
import os
import glob
import kornia as K
from kornia.feature import LoFTR
from tqdm import tqdm

# ==========================================
# 1. LoFTR 特徵匹配函式
# ==========================================
def match_loftr(img1, img2, device):
    """
    使用 LoFTR 進行深度學習特徵匹配
    """
    # 縮放圖片以符合顯示記憶體 (RTX 5060 Ti)
    MAX_SIZE = 840 
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    scale1 = MAX_SIZE / max(h1, w1)
    scale2 = MAX_SIZE / max(h2, w2)
    
    img1_res = cv2.resize(img1, (int(w1 * scale1), int(h1 * scale1)))
    img2_res = cv2.resize(img2, (int(w2 * scale2), int(h2 * scale2)))
    
    gray1 = cv2.cvtColor(img1_res, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_res, cv2.COLOR_BGR2GRAY)
    
    # 轉換為 Tensor 並搬移至 GPU (CUDA)
    t1 = K.image_to_tensor(gray1, False).float() / 255.
    t2 = K.image_to_tensor(gray2, False).float() / 255.
    t1, t2 = t1.to(device), t2.to(device)
    
    # 載入預訓練模型
    matcher = LoFTR(pretrained='outdoor').to(device)
    
    with torch.inference_mode():
        input_dict = {"image0": t1, "image1": t2}
        correspondences = matcher(input_dict)
        
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    
    # 座標還原回原始解析度
    return mkpts0 / scale1, mkpts1 / scale2

# ==========================================
# 2. SPW (Single-Perspective Warps) 縫合類別
# ==========================================
class SPWStitcher:
    """
    針對狹窄場域與大視差設計的縫合類別
    """
    def __init__(self, grid_size=(40, 40), sigma=100):
        self.grid_size = grid_size
        self.sigma = sigma

    def Moving_DLT(self, grid_center, mkpts1, mkpts2):
        """
        APAP 核心：計算局部單應性矩陣
        """
        # 這裡目前以全域 H 作為預設，未來需實作加權 SVD 優化
        H, _ = cv2.findHomography(mkpts1, mkpts2, cv2.RANSAC, 5.0)
        return H if H is not None else np.eye(3)

    def calculate_spw_transform(self, img_up, img_down, mkpts_up, mkpts_down):
        h_up, w_up = img_up.shape[:2]
        h_down, w_down = img_down.shape[:2]

        global_H, _ = cv2.findHomography(mkpts_up, mkpts_down, cv2.RANSAC, 5.0)
        if global_H is None: return None

        corners_up = np.float32([[0, 0], [0, h_up], [w_up, h_up], [w_up, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners_up, global_H)
        corners_down = np.float32([[0, 0], [0, h_down], [w_down, h_down], [w_down, 0]]).reshape(-1, 1, 2)
        
        all_pts = np.concatenate((warped_corners, corners_down), axis=0)
        x_min, y_min = np.int32(all_pts.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_pts.max(axis=0).ravel() + 0.5)
        
        canvas_w = x_max - x_min
        canvas_h = y_max - y_min

        # --- 加入安全門檻 ---
        # 正常番茄植株縫合後的尺寸通常不會超過原始尺寸的 3-5 倍
        MAX_DIMENSION = 12000 
        if canvas_w > MAX_DIMENSION or canvas_h > MAX_DIMENSION:
            print(f"⚠️ 警告: 畫布尺寸過大 ({canvas_w}x{canvas_h})，可能是矩陣運算崩潰。")
            return None
        # ------------------
        
        trans_dist = [-x_min, -y_min]
        H_trans = np.array([[1, 0, trans_dist[0]], [0, 1, trans_dist[1]], [0, 0, 1]])
        
        return global_H, H_trans, (canvas_w, canvas_h)



    def stitch(self, img_up, img_down, H, H_trans, canvas_size):
        """
        執行最終影像合成
        """
        # 執行 Up.jpg 變形
        warped_up = cv2.warpPerspective(img_up, H_trans @ H, canvas_size)
        
        # 貼上 Down.jpg
        ty, tx = int(H_trans[1, 2]), int(H_trans[0, 2])
        h_d, w_d = img_down.shape[:2]
        
        # 建立最終結果 (處理重疊區域)
        result = warped_up.copy()
        result[ty:ty+h_d, tx:tx+w_d] = img_down
        
        return result

# ==========================================
# 3. 主執行流程
# ==========================================
def main():
    # 修正路徑：指向鄰居資料夾 images
    BASE_IMAGE_DIR = "../images/20251205"
    RESULT_DIR = "Result_SPW"
    
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 啟動系統 | 使用裝置: {device}")

    stitcher = SPWStitcher()
    plant_folders = sorted(glob.glob(os.path.join(BASE_IMAGE_DIR, "plant_*")))
    print(f"🔍 找到 {len(plant_folders)} 個待處理植株資料夾\n")

    for folder in plant_folders:
        plant_name = os.path.basename(folder)
        up_p = os.path.join(folder, "Up.jpg")
        down_p = os.path.join(folder, "Down.jpg")

        if os.path.exists(up_p) and os.path.exists(down_p):
            img_u, img_d = cv2.imread(up_p), cv2.imread(down_p)
            
            # LoFTR 匹配
            print(f"[{plant_name}] 正在匹配特徵點...")
            pts_u, pts_d = match_loftr(img_u, img_d, device)
            
            # SPW 運算
            print(f"[{plant_name}] 正在執行 SPW 空間變換...")
            res_data = stitcher.calculate_spw_transform(img_u, img_d, pts_u, pts_d)
            
            if res_data:
                H, H_t, size = res_data
                stitched = stitcher.stitch(img_u, img_d, H, H_t, size)
                
                out_path = os.path.join(RESULT_DIR, f"{plant_name}_final.jpg")
                cv2.imwrite(out_path, stitched)
                print(f"✅ 完成！儲存至 {out_path}\n")
            else:
                print(f"❌ {plant_name} 矩陣運算失敗\n")

if __name__ == "__main__":
    main()
