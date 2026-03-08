import cv2
import torch
import numpy as np
import os
import kornia as K
from kornia.feature import LoFTR
from tqdm import tqdm

# --- 1. LoFTR 匹配函式 ---
def match_loftr(img1, img2, device):
    matcher = LoFTR(pretrained='outdoor').to(device)
    MAX_SIZE = 840
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    s1, s2 = MAX_SIZE / max(h1, w1), MAX_SIZE / max(h2, w2)
    
    t1 = K.image_to_tensor(cv2.resize(img1, (int(w1*s1), int(h1*s1))), False).float().to(device) / 255.
    t2 = K.image_to_tensor(cv2.resize(img2, (int(w2*s2), int(h2*s2))), False).float().to(device) / 255.
    
    with torch.inference_mode():
        out = matcher({"image0": K.color.rgb_to_grayscale(t1), "image1": K.color.rgb_to_grayscale(t2)})
        
    return out['keypoints0'].cpu().numpy() / s1, out['keypoints1'].cpu().numpy() / s2

# --- 2. APAP 核心處理類別 ---
class APAPStitcher:
    def __init__(self, grid_res=(40, 40), gamma=1.5, sigma=12.0):
        self.grid_res = grid_res # 網格解析度
        self.gamma = gamma       # 權重平滑參數
        self.sigma = sigma       # APAP 視差權重

    def get_moving_dlt_h(self, grid_pt, mkpts1, mkpts2):
        """計算單一網格中心的局部 H 矩陣"""
        # 計算歐幾里得距離權重
        dist = np.linalg.norm(mkpts1 - grid_pt, axis=1)
        weights = np.exp(-dist**2 / (2 * self.sigma**2))
        weights = np.maximum(weights, 0.01) # 防止權重為 0

        # 組建加權 DLT 方程式 (2n x 9 矩陣)
        A = []
        for i in range(len(mkpts1)):
            x, y = mkpts1[i]
            u, v = mkpts2[i]
            w = weights[i]
            A.append([-w*x, -w*y, -w, 0, 0, 0, w*x*u, w*y*u, w*u])
            A.append([0, 0, 0, -w*x, -w*y, -w, w*x*v, w*y*v, w*v])
        
        A = np.array(A)
        _, _, Vh = np.linalg.svd(A)
        L = Vh[-1, :] / Vh[-1, -1]
        return L.reshape(3, 3)

    def process(self, img_up, img_down, mkpts_up, mkpts_down):
        h_u, w_u = img_up.shape[:2]
        
        # 1. 計算畫布尺寸 (以全域 H 為基準防止溢出)
        H_global, _ = cv2.findHomography(mkpts_up, mkpts_down, cv2.RANSAC, 5.0)
        # 設定畫布 (暫定原圖兩倍寬高)
        canvas_h, canvas_w = h_u * 2, w_u * 2
        offset_h, offset_w = h_u // 2, w_u // 2
        H_trans = np.array([[1, 0, offset_w], [0, 1, offset_h], [0, 0, 1]], dtype=np.float32)

        # 2. 建立變形對應表 (Map)
        map_x = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        map_y = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        
        # 3. 逐網格計算變形
        gw, gh = self.grid_res
        cell_w, cell_h = canvas_w // gw, canvas_h // gh
        
        print("正在計算 APAP 網格變形...")
        for i in tqdm(range(gh)):
            for j in range(gw):
                # 網格中心座標
                grid_center = np.array([j * cell_w + cell_w/2 - offset_w, i * cell_h + cell_h/2 - offset_h])
                
                # 計算局部 H
                H_local = self.get_moving_dlt_h(grid_center, mkpts_up, mkpts_down)
                H_final = H_trans @ H_local
                H_inv = np.linalg.inv(H_final)

                # 填充該網格內的像素對應
                y, x = np.indices((cell_h, cell_w))
                pts = np.stack([x + j*cell_w, y + i*cell_h, np.ones_like(x)], axis=-1).reshape(-1, 3)
                
                # 逆向投影 (Backward Warping)
                proj = pts @ H_inv.T
                proj /= proj[:, 2:3]
                
                map_x[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = proj[:, 0].reshape(cell_h, cell_w)
                map_y[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = proj[:, 1].reshape(cell_h, cell_w)

        # 4. 執行重採樣變形
        warped_up = cv2.remap(img_up, map_x, map_y, cv2.INTER_LINEAR)
        
        # 5. 貼上 Down.jpg
        result = warped_up.copy()
        result[offset_h:offset_h+img_down.shape[0], offset_w:offset_w+img_down.shape[1]] = img_down
        
        return result

# --- 3. 執行測試 ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 請修正路徑
    img_u = cv2.imread("../images/20251205/plant_1/Up.jpg")
    img_d = cv2.imread("../images/20251205/plant_1/Down.jpg")
    
    # 旋轉 180 度 (如果還沒轉正的話)
    # img_d = cv2.rotate(img_d, cv2.ROTATE_180) 

    pts_u, pts_d = match_loftr(img_u, img_d, device)
    
    apap = APAPStitcher(grid_res=(50, 50), sigma=12.0)
    final_img = apap.process(img_u, img_d, pts_u, pts_d)
    
    cv2.imwrite("plant_1_APAP_result.jpg", final_img)
    print("✅ APAP 縫合完成！")
