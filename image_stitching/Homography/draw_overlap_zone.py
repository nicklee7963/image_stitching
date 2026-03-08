import cv2
import torch
import numpy as np
import os
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

class OverlapVisualizer:
    def __init__(self, device='cuda'):
        self.device = device
        # 使用目前最強的 LightGlue 進行匹配
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        self.matcher = LightGlue(features='superpoint').eval().to(device)

    def get_overlap_polygon(self, kpts):
        """
        計算特徵點的凸包 (Convex Hull)，這就是重疊區域的邊界
        """
        if len(kpts) < 3: return None
        hull = cv2.convexHull(kpts.astype(np.int32))
        return hull

    def draw_zone(self, image, hull, color=(0, 255, 0)):
        """
        在圖片上畫出半透明的重疊區域
        """
        overlay = image.copy()
        if hull is not None:
            # 畫出多邊形邊界
            cv2.drawContours(overlay, [hull], -1, color, 2)
            # 填滿半透明色彩
            cv2.fillPoly(overlay, [hull], color)
        
        # 混合原圖與遮罩
        return cv2.addWeighted(overlay, 0.3, image, 0.7, 0)

    def process(self, path0, path1):
        # 1. 載入圖片 (Tensor 用於匹配, BGR 用於繪圖)
        image0_t = load_image(path0)
        image1_t = load_image(path1)
        img0_bgr = cv2.imread(path0)
        img1_bgr = cv2.imread(path1)

        # 2. 執行匹配
        feats0 = self.extractor.extract(image0_t.to(self.device))
        feats1 = self.extractor.extract(image1_t.to(self.device))
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        kpts0, kpts1 = feats0['keypoints'], feats1['keypoints']
        matches = matches01['matches']
        
        mkpts0 = kpts0[matches[..., 0]].cpu().numpy()
        mkpts1 = kpts1[matches[..., 1]].cpu().numpy()

        # 3. 計算並繪製重疊區域 (凸包)
        hull0 = self.get_overlap_polygon(mkpts0)
        hull1 = self.get_overlap_polygon(mkpts1)

        res0 = self.draw_zone(img0_bgr, hull0, color=(0, 255, 255)) # 黃色代表 Up 的重疊區
        res1 = self.draw_zone(img1_bgr, hull1, color=(255, 255, 0)) # 藍色代表 Down 的重疊區

        return res0, res1

# ==========================================
# 執行繪圖
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
visualizer = OverlapVisualizer(device)

# 請更換為你的番茄圖片路徑
img_up_path = "/home/nicklee/ssl/image_stitching_learning/image_stitching/images/20251205/plant_1/Up.jpg"
img_down_path = "/home/nicklee/ssl/image_stitching_learning/image_stitching/images/20251205/plant_1/Down.jpg"

res_up, res_down = visualizer.process(img_up_path, img_down_path)

# 儲存結果
cv2.imwrite("Up_Overlap_Zone.jpg", res_up)
cv2.imwrite("Down_Overlap_Zone.jpg", res_down)
print("✅ 重疊區域劃定完成！請查看 Up_Overlap_Zone.jpg 與 Down_Overlap_Zone.jpg")
