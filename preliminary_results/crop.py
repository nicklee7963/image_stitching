import cv2
import torch
import kornia
from kornia.feature import LoFTR
import numpy as np
import os

# ==========================================
# 方法 A: 寬鬆版 LoFTR (Relaxed Mode)
# ==========================================
def run_relaxed_loftr(img1, img2):
    print("--- [方法 A] 嘗試寬鬆版 LoFTR ---")
    
    # 轉灰階並縮圖
    infer_w, infer_h = 640, 480
    input1 = cv2.resize(img1, (infer_w, infer_h))
    input2 = cv2.resize(img2, (infer_w, infer_h))
    
    img1_tensor = kornia.image_to_tensor(cv2.cvtColor(input1, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    img2_tensor = kornia.image_to_tensor(cv2.cvtColor(input2, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained='outdoor').to(device)
    
    with torch.no_grad():
        res = matcher({"image0": img1_tensor.to(device), "image1": img2_tensor.to(device)})

    mkpts0 = res['keypoints0'].cpu().numpy()
    mkpts1 = res['keypoints1'].cpu().numpy()
    conf = res['confidence'].cpu().numpy()

    # --- 關鍵修改 1: 大幅降低信心門檻 ---
    # 只要 > 0.3 就讓它過 (原本是 0.7)
    valid = conf > 0.3
    pts1 = mkpts0[valid]
    pts2 = mkpts1[valid]
    
    print(f"初步找到點數: {len(pts1)}")
    
    if len(pts1) < 4:
        print("點數還是太少，LoFTR 失敗。")
        return None

    # 還原座標
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1[:, 0] *= (w1 / infer_w)
    pts1[:, 1] *= (h1 / infer_h)
    pts2[:, 0] *= (w2 / infer_w)
    pts2[:, 1] *= (h2 / infer_h)

    # --- 關鍵修改 2: 放寬 RANSAC 容忍度 ---
    # ransacReprojThreshold 設為 30.0 (原本是 3.0)
    # 這代表允許 30 像素的誤差 (視差造成的錯位)
    M, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 30.0)
    
    if mask is not None:
        inliers = np.sum(mask)
        print(f"RANSAC 後有效點數: {inliers} (若 > 10 就算成功)")
        
        # 繪製連線圖 (只畫 Inliers)
        vis = np.zeros((max(h1, h2), w1+w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:] = img2
        
        mask = mask.ravel()
        pts1_in = pts1[mask == 1]
        pts2_in = pts2[mask == 1]
        
        # 隨機畫 50 條
        indices = np.arange(len(pts1_in))
        np.random.shuffle(indices)
        for i in indices[:50]:
            p1 = (int(pts1_in[i][0]), int(pts1_in[i][1]))
            p2 = (int(pts2_in[i][0] + w1), int(pts2_in[i][1]))
            cv2.line(vis, p1, p2, (0, 255, 0), 2)
            
        cv2.imwrite("result_A_relaxed_loftr.jpg", vis)
        return M
    else:
        print("RANSAC 計算失敗。")
        return None

# ==========================================
# 方法 B: 模板匹配 (Template Matching) - 保底方案
# ==========================================
def run_template_matching(img1, img2):
    print("\n--- [方法 B] 嘗試模板匹配 (暴力拼接) ---")
    # 假設 img1 是上半部，img2 是下半部
    # 我們切下 img1 最底部的一條 (作為模板)
    h1, w1 = img1.shape[:2]
    
    template_h = 100 # 取底部 100 pixel
    template = img1[h1-template_h:h1, 50:w1-50] # 左右內縮一點避免黑邊
    
    # 在 img2 的上半部尋找這個模板
    # 限制搜尋範圍：img2 的上半部 (假設重疊區在上面)
    search_h = int(img2.shape[0] * 0.7)
    target = img2[:search_h, :]
    
    # 執行匹配
    res = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    print(f"匹配信心度: {max_val:.2f}")
    
    if max_val < 0.3:
        print("模板匹配失敗，重疊特徵不明顯。")
        return None
        
    # max_loc 是模板在 target 中的左上角座標 (x, y)
    # 這代表 img2 的這個位置 = img1 的底部 (h1 - template_h)
    # 所以 img2 相對於 img1 的位移:
    # shift_y = (h1 - template_h) - match_y
    match_x, match_y = max_loc
    
    # 計算 img2 應該貼在哪裡 (相對於 img1 左上角)
    # img1 的底部 (h1) 應該對齊 img2 的 (match_y + template_h)
    # 也就是 img2 的頂部 (0) 應該在 img1 的 (h1 - (match_y + template_h))
    
    offset_y = h1 - (match_y + template_h)
    offset_x = 0 - (match_x - 50) # 修正 x 偏移 (假設只有垂直移動，這項通常很小)
    
    print(f"計算出位移: y={offset_y}, x={offset_x}")
    
    # 建立平移矩陣
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    return M

# ==========================================
# 縫合工具
# ==========================================
def simple_stitch(img1, img2, M, filename):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 預估畫布大小 (往下接)
    canvas_h = int(h1 + h2) 
    canvas_w = max(w1, w2)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # 貼上 img1
    canvas[:h1, :w1] = img1
    
    # 貼上 img2
    warped_img2 = cv2.warpAffine(img2, M, (canvas_w, canvas_h))
    
    # 簡單覆蓋 (非黑色區域)
    mask = np.any(warped_img2 != [0,0,0], axis=-1)
    canvas[mask] = warped_img2[mask]
    
    # 裁切多餘黑邊
    rows = np.any(canvas != [0,0,0], axis=(1, 2))
    cols = np.any(canvas != [0,0,0], axis=(0, 2))
    if np.any(rows) and np.any(cols):
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        canvas = canvas[rmin:rmax+1, cmin:cmax+1]
    
    cv2.imwrite(filename, canvas)
    print(f"縫合完成，已儲存: {filename}")

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    # 使用你裁切好的上下兩張圖 (確保 img1 是上，img2 是下)
    img1_path = "image1_cropped.jpg" # 請換成你實際的檔名
    img2_path = "image2_cropped.jpg" # 請換成你實際的檔名
    
    # 如果你沒有存裁切圖，可以直接讀原圖並在這裡裁 (取消註解下面這段)
    """
    full_img1 = cv2.imread("image1.jpg")
    full_img2 = cv2.imread("image2.jpg")
    h, w = full_img1.shape[:2]
    img1_path = None
    img1 = full_img1[int(h*0.3):, :] # 取圖1下半
    img2 = full_img2[:int(h*0.7), :] # 取圖2上半
    """
    
    # 讀取 (如果你已經有裁切好的檔案)
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
    else:
        # 為了測試，先讀原圖並自動裁切 (假設)
        print("讀取原圖進行自動裁切...")
        raw1 = cv2.imread("image1.jpg")
        raw2 = cv2.imread("image2.jpg")
        if raw1 is not None:
             h, w = raw1.shape[:2]
             # 圖1留下面 60%
             img1 = raw1[int(h*0.4):, :]
             # 圖2留上面 60%
             img2 = raw2[:int(h*0.6), :]
        else:
            print("找不到圖片！")
            exit()

    # --- 1. 跑寬鬆版 LoFTR ---
    M_loftr = run_relaxed_loftr(img1, img2)
    if M_loftr is not None:
        # LoFTR 回傳的是 3x3 Homography，轉成 2x3 Affine 方便顯示
        # 這裡簡化直接用，或者如果你需要 Affine，可以用 estimateAffinePartial2D
        # 為了保險，我們這裡不縫 LoFTR 的結果，只看連線圖 result_A_relaxed_loftr.jpg
        pass 

    # --- 2. 跑模板匹配 (保底) ---
    M_template = run_template_matching(img1, img2)
    if M_template is not None:
        simple_stitch(img1, img2, M_template, "result_B_template_stitch.jpg")
