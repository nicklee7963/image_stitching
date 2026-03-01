import cv2
import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt

def run_sold2(img1_path, img2_path):
    print("\n--- 執行 SOLD2 (線特徵匹配) ---")
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    # 縮圖以免跑太久 (SOLD2 比較慢)
    infer_w, infer_h = 800, 600
    input1 = cv2.resize(img1, (infer_w, infer_h))
    input2 = cv2.resize(img2, (infer_w, infer_h))
    
    img1_t = kornia.image_to_tensor(input1, False).float() / 255.0
    img2_t = kornia.image_to_tensor(input2, False).float() / 255.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用 Kornia 內建的 SOLD2
    sold2 = kornia.feature.SOLD2(pretrained=True, config=None).to(device)
    
    with torch.no_grad():
        # 偵測與描述
        out1 = sold2(img1_t.to(device))
        out2 = sold2(img2_t.to(device))
        
        # 匹配
# 修正後的順序：先放兩張圖的線 (seg1, seg2)，再放兩張圖的特徵 (desc1, desc2)
# 修正重點：
        # 1. 線段 (line_segments) 要取 [0]，因為它是變長的 List。
        # 2. 特徵 (dense_desc) 不要取 [0]，必須保留 [Batch, Channel, H, W] 的 4D 格式。
        matches = sold2.match(out1["line_segments"][0], out2["line_segments"][0],
                              out1["dense_desc"], out2["dense_desc"])
    line_seg1 = out1["line_segments"][0].cpu().numpy()
    line_seg2 = out2["line_segments"][0].cpu().numpy()
    matches = matches.cpu().numpy()
    
    valid_matches = matches != -1
    match_indices = np.where(valid_matches)[0]
    
    print(f"SOLD2 找到匹配線段數: {len(match_indices)}")
    
    # --- 繪圖 ---
    # 建立畫布
    h, w = input1.shape
    vis = np.zeros((h, w*2), dtype=np.uint8)
    vis[:, :w] = input1
    vis[:, w:] = input2
    
    # 轉成彩色以便畫線
    vis_color = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    # 畫線
    # 隨機選 30 條畫就好，不然會太亂
    if len(match_indices) > 30:
        draw_indices = np.random.choice(match_indices, 30, replace=False)
    else:
        draw_indices = match_indices
        
    for idx in draw_indices:
        # 左圖線段
        l1 = line_seg1[idx]
        p1_start = (int(l1[0, 1]), int(l1[0, 0])) # SOLD2 輸出是 (y, x)
        p1_end = (int(l1[1, 1]), int(l1[1, 0]))
        
        # 右圖線段 (對應的 index)
        idx2 = matches[idx]
        l2 = line_seg2[idx2]
        p2_start = (int(l2[0, 1] + w), int(l2[0, 0]))
        p2_end = (int(l2[1, 1] + w), int(l2[1, 0]))
        
        # 畫線
        color = (0, 255, 255) # 黃色
        cv2.line(vis_color, p1_start, p1_end, color, 2)
        cv2.line(vis_color, p2_start, p2_end, color, 2)
        # 連接兩圖
        cv2.line(vis_color, p1_start, p2_start, (255, 0, 0), 1)

    cv2.imwrite("result_3_SOLD2_lines.jpg", vis_color)
    print("SOLD2 結果已儲存: result_3_SOLD2_lines.jpg")

if __name__ == "__main__":
    run_sold2("image1.jpg", "image2.jpg")
