import cv2
import torch
import kornia
from kornia.feature import LoFTR
import numpy as np
import os

# ==========================================
# 1. 自動生成重疊測試圖
# ==========================================
def generate_synthetic_pairs(image_path, overlap_ratio=0.5):
    print(f"--- 正在生成模擬測試圖 (重疊率: {overlap_ratio:.0%}) ---")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"找不到圖片: {image_path}")
    
    h, w = img.shape[:2]
    
    # 我們模擬「垂直掃描」：一張在上，一張在下
    # 兩張圖的高度各佔原圖的 60% (所以中間會重疊 20%)
    # 或者依照你的要求，重疊多一點，設 crop_h = 75% -> 重疊 50%
    
    crop_h = int(h * 0.75) # 每張圖高佔 75%
    
    # Top Image (上半部)
    # 取範圍: [0 : crop_h]
    img_top = img[:crop_h, :]
    
    # Bottom Image (下半部)
    # 取範圍: [h - crop_h : h]
    # 起始點計算: h - crop_h
    start_y = h - crop_h
    img_bottom = img[start_y:, :]
    
    print(f"原圖尺寸: {w}x{h}")
    print(f"生成上半部: {w}x{crop_h}")
    print(f"生成下半部: {w}x{crop_h}")
    print(f"預期垂直位移 (Ground Truth): {start_y} pixels")
    
    return img_top, img_bottom, start_y

# ==========================================
# 2. LoFTR 匹配與驗證
# ==========================================
def run_loftr_test(img1, img2, gt_shift_y):
    print("\n--- 執行 LoFTR 匹配驗證 ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained='outdoor').to(device)

    # 縮圖計算
    infer_w, infer_h = 640, 480
    img1_tensor = kornia.image_to_tensor(cv2.cvtColor(cv2.resize(img1, (infer_w, infer_h)), cv2.COLOR_BGR2GRAY), False).float() / 255.0
    img2_tensor = kornia.image_to_tensor(cv2.cvtColor(cv2.resize(img2, (infer_w, infer_h)), cv2.COLOR_BGR2GRAY), False).float() / 255.0
    
    with torch.no_grad():
        res = matcher({"image0": img1_tensor.to(device), "image1": img2_tensor.to(device)})

    mkpts0 = res['keypoints0'].cpu().numpy()
    mkpts1 = res['keypoints1'].cpu().numpy()
    conf = res['confidence'].cpu().numpy()

    # 過濾
    valid = conf > 0.5
    mkpts0 = mkpts0[valid]
    mkpts1 = mkpts1[valid]
    
    # 座標還原
    scale_x = img1.shape[1] / infer_w
    scale_y = img1.shape[0] / infer_h
    
    mkpts0[:, 0] *= scale_x
    mkpts0[:, 1] *= scale_y
    mkpts1[:, 0] *= scale_x
    mkpts1[:, 1] *= scale_y

    print(f"LoFTR 找到高信心點數: {len(mkpts0)}")

    # --- 驗證準確度 ---
    # 理論上，因為是從同一張圖切出來的，且沒有左右移動
    # 所以 pts1 (上圖點) 的 y 座標 + gt_shift_y 應該要等於 pts2 (下圖點) 的 y 座標？
    # 不對，因為我們是把「下半部」當作 img2。
    # img2 的 (0,0) 其實是原圖的 (start_y, 0)
    # 所以 img2 裡的某個點 (x, y)，在 img1 (上半部，原點也是0,0) 裡的對應點應該是 (x, y + start_y)
    # 也就是: pts1.y = pts2.y + shift (如果是完美的)
    
    # 我們算一下平均誤差
    # 這裡的邏輯稍微繞一下：
    # img1 是 top crop (範圍 0~750)
    # img2 是 bottom crop (範圍 250~1000)
    # 重疊區是 250~750
    # img2 的 y=0 對應原圖 y=250
    # img1 的 y=250 對應原圖 y=250
    # 所以匹配點應該滿足: pts1.y = pts2.y + gt_shift_y
    
    diff_y = mkpts0[:, 1] - mkpts1[:, 1]
    estimated_shift = np.median(diff_y)
    
    print(f"Ground Truth 位移: {gt_shift_y:.2f}")
    print(f"LoFTR 估計位移: {estimated_shift:.2f}")
    print(f"誤差: {abs(estimated_shift - gt_shift_y):.2f} pixels")

    # 繪製連線圖
    h, w = img1.shape[:2]
    vis_img = np.zeros((h, w*2, 3), dtype=np.uint8)
    vis_img[:h, :w] = img1
    vis_img[:h, w:] = img2
    
    # 隨機畫 50 條線
    indices = np.arange(len(mkpts0))
    np.random.shuffle(indices)
    for i in indices[:50]:
        pt1 = (int(mkpts0[i][0]), int(mkpts0[i][1]))
        pt2 = (int(mkpts1[i][0] + w), int(mkpts1[i][1]))
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)

    cv2.imwrite("synthetic_matches.jpg", vis_img)
    print("已儲存連線圖: synthetic_matches.jpg")

    # --- 執行簡易縫合 (驗證用) ---
    # 我們試著把 img2 貼回 img1 下面
    # 建立大畫布 (原圖大小)
    # 總高度 = gt_shift_y + img2.height
    total_h = int(estimated_shift + img2.shape[0])
    stitch_canvas = np.zeros((total_h, w, 3), dtype=np.uint8)
    
    # 貼上 img1
    stitch_canvas[:img1.shape[0], :] = img1
    
    # 貼上 img2 (融合)
    # 我們簡單做：把 img2 貼到它該在的位置
    y_offset = int(estimated_shift)
    
    # 為了看清楚接縫，我們用半透明混合重疊區
    # 非重疊區直接覆蓋
    stitch_canvas[y_offset:, :] = img2
    
    cv2.imwrite("synthetic_stitch.jpg", stitch_canvas)
    print("已儲存縫合圖: synthetic_stitch.jpg")

if __name__ == "__main__":
    # 使用你上傳的其中一張清晰的大圖
    # 請確保 image1.jpg 是那張高解析度的
    run_image = "image1.jpg" 
    
    if os.path.exists(run_image):
        img_top, img_bot, shift = generate_synthetic_pairs(run_image, overlap_ratio=0.5)
        run_loftr_test(img_top, img_bot, shift)
    else:
        print("找不到圖片，請確認檔名。")
