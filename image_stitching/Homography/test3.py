import cv2
import torch
import numpy as np
import os
import glob
import kornia as K
from kornia.feature import LoFTR
from tqdm import tqdm

# ==========================================
# 1. 3D 幾何恢復模組 (模擬 PIS3R 步驟 1)
# ==========================================
def match_loftr(img1, img2, device):
    matcher = LoFTR(pretrained='outdoor').to(device)
    # 針對 RTX 5060 Ti 優化解析度
    MAX_SIZE = 840 
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    s1, s2 = MAX_SIZE / max(h1, w1), MAX_SIZE / max(h2, w2)
    
    t1 = K.image_to_tensor(cv2.resize(img1, (int(w1*s1), int(h1*s1))), False).float().to(device) / 255.
    t2 = K.image_to_tensor(cv2.resize(img2, (int(w2*s2), int(h2*s2))), False).float().to(device) / 255.
    
    with torch.inference_mode():
        out = matcher({"image0": K.color.rgb_to_grayscale(t1), "image1": K.color.rgb_to_grayscale(t2)})
        
    return out['keypoints0'].cpu().numpy() / s1, out['keypoints1'].cpu().numpy() / s2

# ==========================================
# 2. PIS3R 概念縫合類別
# ==========================================
class PIS3RConceptStitcher:
    def __init__(self):
        # 假設相機內參 (這部分在專題中建議使用真實相機校正值)
        # 這裡根據一般手機/工業相機視角預估
        self.K = np.array([
            [1200, 0, 420], 
            [0, 1200, 315], 
            [0, 0, 1]
        ], dtype=np.float32)

    def estimate_3d_pose(self, pts1, pts2):
        """
        從 2D 點對恢復 3D 相機位姿 (R, t)
        """
        # 1. 計算基礎矩陣 (Essential Matrix)
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None: return None, None
        
        # 2. 分解出旋轉矩陣 R 與 平移向量 t
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        return R, t

    def get_stitching_geometry(self, img_up, img_down, R, t):
        """
        執行 3D 重投影對齊
        """
        h, w = img_up.shape[:2]
        
        # 這裡我們使用平面誘導單應性 (Plane-induced Homography)
        # 假設番茄植株主要落在一個垂直平面上 (距離 d=0.3公尺)
        d = 0.3
        normal = np.array([0, 0, 1]).reshape(3, 1) # 假設垂直於鏡頭
        
        # 3D 投影公式：H = K * (R + t*n/d) * inv(K)
        H_3d = self.K @ (R + (t @ normal.T) / d) @ np.linalg.inv(self.K)
        
        # 設置畫布 (避免爆炸)
        canvas_h, canvas_w = int(h * 2.2), int(w * 1.5)
        offset_x, offset_y = w // 4, h // 2
        H_trans = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)
        
        return H_3d, H_trans, (canvas_w, canvas_h)

    def run(self, img_up, img_down, pts_up, pts_down):
        # 1. 估算 3D 幾何
        R, t = self.estimate_3d_pose(pts_up, pts_down)
        if R is None: return None
        
        # 2. 計算 3D 投影縫合矩陣
        H_3d, H_trans, size = self.get_stitching_geometry(img_up, img_down, R, t)
        
        # 3. 變形與融合 (Warping)
        warped_up = cv2.warpPerspective(img_up, H_trans @ H_3d, size)
        
        # 貼上 Down.jpg
        ty, tx = int(H_trans[1, 2]), int(H_trans[0, 2])
        h_d, w_d = img_down.shape[:2]
        
        result = warped_up.copy()
        result[ty:ty+h_d, tx:tx+w_d] = img_down
        
        return result

# ==========================================
# 3. 主執行流程
# ==========================================
def main():
    BASE_DIR = "../images/20251205"
    OUT_DIR = "Result_PIS3R_Concept"
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 PIS3R 概念系統啟動 | 裝置: {device}")

    stitcher = PIS3RConceptStitcher()
    folders = sorted(glob.glob(os.path.join(BASE_DIR, "plant_*")))

    for folder in folders:
        name = os.path.basename(folder)
        up_path = os.path.join(folder, "Up.jpg")
        down_path = os.path.join(folder, "Down.jpg")

        if os.path.exists(up_path) and os.path.exists(down_path):
            img_u, img_d = cv2.imread(up_path), cv2.imread(down_path)
            
            print(f"[{name}] 執行 3D 幾何恢復...")
            try:
                # LoFTR 匹配
                pts_u, pts_d = match_loftr(img_u, img_d, device)
                
                # 執行基於 3D 重建邏輯的縫合
                res = stitcher.run(img_u, img_d, pts_u, pts_d)
                
                if res is not None:
                    cv2.imwrite(os.path.join(OUT_DIR, f"{name}_3D.jpg"), res)
                    print(f"✅ {name} 3D 重投影縫合完成！\n")
                else:
                    print(f"❌ {name} 幾何恢復失敗。\n")
            except Exception as e:
                print(f"💥 錯誤: {e}\n")

if __name__ == "__main__":
    main()
