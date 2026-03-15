import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torch
import kornia as K
import kornia.feature as KF

# ==========================================
# 1. 初始化與環境設定
# ==========================================
IMG_PATH = "../Images/20251205/plant_1/Up.jpg"
RESULT_DIR = "./result"
GLOBAL_CSV = os.path.join(RESULT_DIR, "result.csv")

os.makedirs(RESULT_DIR, exist_ok=True)
next_idx = len([d for d in os.listdir(RESULT_DIR) if d.startswith('test_')]) + 1
CURRENT_TEST_DIR = os.path.join(RESULT_DIR, f"test_{next_idx:02d}")
os.makedirs(CURRENT_TEST_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 讀取並統一解析度為 840px (公平基準)
img_orig_raw = cv2.imread(IMG_PATH)
if img_orig_raw is None:
    print(f"錯誤：找不到圖片 {IMG_PATH}"); exit()

h_raw, w_raw = img_orig_raw.shape[:2]
MAX_DIM = 840
scale_factor = MAX_DIM / max(h_raw, w_raw)
img_orig = cv2.resize(img_orig_raw, (int(w_raw * scale_factor), int(h_raw * scale_factor)))

# ==========================================
# 2. 視覺化除錯工具
# ==========================================
def draw_matches(img1, img2, pts1, pts2, name, test_dir):
    """將特徵配對連線畫出來並存檔，用來抓出作弊行為"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2
    
    # 最多只畫前 80 個點避免畫面太亂
    for p1, p2 in zip(pts1[:80], pts2[:80]):
        pt1 = (int(p1[0][0]), int(p1[0][1]))
        pt2 = (int(p2[0][0]) + w1, int(p2[0][1]))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(canvas, pt1, 5, color, -1)
        cv2.circle(canvas, pt2, 5, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1)
        
    cv2.imwrite(os.path.join(test_dir, f"{name}_matches.jpg"), canvas)

# ==========================================
# 3. 論文光影與幾何生成
# ==========================================
def apply_random_variety(img):
    """產生 Augmented Image (基準圖)"""
    alpha = np.random.uniform(0.6, 1.4) 
    beta = np.random.randint(-40, 40)
    img_aug = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    if np.random.rand() > 0.3:
        img_aug = cv2.GaussianBlur(img_aug, (9, 9), 0)
        
    h, w = img_aug.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    cx, cy = np.random.randint(0, w), np.random.randint(0, h)
    cv2.circle(mask, (cx, cy), np.random.randint(150, 250), 0.3, -1)
    mask = cv2.GaussianBlur(mask, (101, 101), 0)
    return (img_aug * mask[:, :, np.newaxis]).astype(np.uint8)

def get_paper_spec_h(shape):
    """產生 Ground Truth H"""
    h, w = shape[:2]
    cx, cy = w / 2, h / 2
    
    angle = np.random.uniform(-40, 40)
    scale = np.random.uniform(0.8, 1.2)
    M_rot_scale = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    H_rot_scale = np.vstack([M_rot_scale, [0, 0, 1]])
    
    tx, ty = np.random.uniform(-w*0.05, w*0.05), np.random.uniform(-h*0.05, h*0.05)
    H_trans = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    
    shx, shy = np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)
    H_shear = np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]])
    
    px, py = np.random.uniform(-0.0001, 0.0001), np.random.uniform(-0.0001, 0.0001)
    H_persp = np.array([[1, 0, 0], [0, 1, 0], [px, py, 1]])
    
    H = H_persp @ H_trans @ H_shear @ H_rot_scale
    return H / H[2, 2]

# ==========================================
# 4. 核心演算法 (加入防作弊與尺寸修復)
# ==========================================
def match_sift_orb(img_clean, img_dirty, mode="SIFT", test_dir=None):
    det = cv2.SIFT_create() if mode == "SIFT" else cv2.ORB_create(3000)
    
    # 【關鍵防作弊】：過濾 Warp 產生的純黑邊界，強制 SIFT 在圖內找特徵
    gray2 = cv2.cvtColor(img_dirty, cv2.COLOR_BGR2GRAY)
    _, mask2 = cv2.threshold(gray2, 1, 255, cv2.THRESH_BINARY)
    mask2 = cv2.erode(mask2, np.ones((15, 15), np.uint8), iterations=2)
    
    kp1, des1 = det.detectAndCompute(img_clean, None)
    kp2, des2 = det.detectAndCompute(img_dirty, mask2)
    
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4: 
        return np.eye(3)
    
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50)) if mode == "SIFT" else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    if mode == "SIFT":
        matches = matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    else:
        matches = matcher.match(des1, des2)
        good = sorted(matches, key=lambda x: x.distance)[:100]
    
    if len(good) > 10:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        if test_dir: draw_matches(img_clean, img_dirty, pts1, pts2, mode, test_dir)
        H_est, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
        return H_est if H_est is not None else np.eye(3)
    return np.eye(3)

def match_loftr(img_clean, img_dirty, test_dir=None):
    matcher = KF.LoFTR(pretrained='outdoor').to(device).eval()
    
    h, w = img_clean.shape[:2]
    new_h, new_w = (h // 8) * 8, (w // 8) * 8
    img1_8 = cv2.resize(img_clean, (new_w, new_h))
    img2_8 = cv2.resize(img_dirty, (new_w, new_h))
    
    t1 = K.image_to_tensor(cv2.cvtColor(img1_8, cv2.COLOR_BGR2GRAY), False).float().to(device)/255.
    t2 = K.image_to_tensor(cv2.cvtColor(img2_8, cv2.COLOR_BGR2GRAY), False).float().to(device)/255.
    
    with torch.no_grad():
        out = matcher({"image0": t1, "image1": t2})
    
    pts1_raw = out['keypoints0'].cpu().numpy()
    pts2_raw = out['keypoints1'].cpu().numpy()
    
    # === 關鍵修復：取得信心分數並由高到低排序 ===
    conf = out['confidence'].cpu().numpy()
    sort_idx = np.argsort(conf)[::-1] # 取得降冪排序的索引
    pts1_raw = pts1_raw[sort_idx]
    pts2_raw = pts2_raw[sort_idx]
    # ============================================

    pts1 = np.zeros_like(pts1_raw)
    pts1[:, 0] = pts1_raw[:, 0] * (w / new_w)
    pts1[:, 1] = pts1_raw[:, 1] * (h / new_h)
    
    pts2 = np.zeros_like(pts2_raw)
    pts2[:, 0] = pts2_raw[:, 0] * (w / new_w)
    pts2[:, 1] = pts2_raw[:, 1] * (h / new_h)
    
    if len(pts1) > 10:
        pts1_reshaped = np.float32(pts1).reshape(-1, 1, 2)
        pts2_reshaped = np.float32(pts2).reshape(-1, 1, 2)
        
        # 現在傳給畫圖函式的，會是最有信心的前 80 個點了！
        if test_dir: draw_matches(img_clean, img_dirty, pts1_reshaped, pts2_reshaped, "LoFTR", test_dir)
        
        H_est, _ = cv2.findHomography(pts1_reshaped, pts2_reshaped, cv2.RANSAC, 4.0)
        return H_est if H_est is not None else np.eye(3)
    return np.eye(3)

# ==========================================
# 5. 論文指標評估 (Wrapped vs Transformed)
# ==========================================
def evaluate_paper_metrics(H_gt, H_est, img_aug, img_trans):
    if np.array_equal(H_est, np.eye(3)):
        return 500.0, 0.0, 255.0**2, 0.0, np.zeros_like(img_trans)

    H_gt_n = H_gt / H_gt[2, 2]
    H_est_n = H_est / H_est[2, 2]
    
    frob = np.linalg.norm(H_gt_n - H_est_n, ord='fro')
    v1, v2 = H_gt_n.flatten(), H_est_n.flatten()
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    h, w = img_aug.shape[:2]
    try:
        img_wrapped = cv2.warpPerspective(img_aug, H_est_n, (w, h))
    except:
        img_wrapped = np.zeros_like(img_aug)
        
    gray_trans = cv2.cvtColor(img_trans, cv2.COLOR_BGR2GRAY)
    gray_wrapped = cv2.cvtColor(img_wrapped, cv2.COLOR_BGR2GRAY)
    
    mse_val = np.mean((gray_trans.astype(np.float32) - gray_wrapped.astype(np.float32)) ** 2)
    ssim_val = ssim(gray_trans, gray_wrapped, data_range=255)
    
    return frob, cos_sim, mse_val, ssim_val, img_wrapped

# ==========================================
# 6. 主程式執行
# ==========================================
print(f"🚀 開始第 {next_idx} 次測試...")

# A. 產生資料
img_aug = apply_random_variety(img_orig)
H_gt = get_paper_spec_h(img_orig.shape)
h_orig, w_orig = img_orig.shape[:2]
img_trans = cv2.warpPerspective(img_aug, H_gt, (w_orig, h_orig))

results = []
for name in ["SIFT", "ORB", "LoFTR"]:
    print(f"正在執行 {name}...")
    
    # B. 核心配對 (傳入 test_dir 以便畫出連線圖)
    H_est = match_loftr(img_orig, img_trans, CURRENT_TEST_DIR) if name == "LoFTR" else match_sift_orb(img_orig, img_trans, name, CURRENT_TEST_DIR)
    
    # C. 計算指標
    frob, cos, mse, s_val, img_wrapped = evaluate_paper_metrics(H_gt, H_est, img_aug, img_trans)
    
    # D. 視覺化存檔
    cv2.putText(img_trans, "Transformed (GT)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    img_wrapped_viz = img_wrapped.copy()
    cv2.putText(img_wrapped_viz, f"{name} Wrapped", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    canvas = np.hstack((img_trans, img_wrapped_viz))
    cv2.imwrite(os.path.join(CURRENT_TEST_DIR, f"{name}_vs_GT.jpg"), canvas)
    
    results.append({"Method": name, "MSE": mse, "SSIM": s_val, "Frobenius": frob, "Cosine": cos})

# ==========================================
# 7. 數據儲存與繪圖
# ==========================================
df = pd.DataFrame(results)
df['Test_ID'] = f"test_{next_idx:02d}"
df.to_csv(GLOBAL_CSV, mode='a', header=not os.path.exists(GLOBAL_CSV), index=False)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
metrics = [('MSE','skyblue'), ('SSIM','salmon'), ('Frobenius','green'), ('Cosine','gold')]

for i, (m, c) in enumerate(metrics):
    df.plot(x='Method', y=m, kind='bar', ax=axes[i//2, i%2], color=c, rot=0, title=m)
    if m == 'Cosine':
        min_cos = df[df['Cosine'] > 0]['Cosine'].min()
        axes[i//2, i%2].set_ylim(max(0, min_cos - 0.05), 1.005) 

plt.tight_layout()
plt.savefig(os.path.join(CURRENT_TEST_DIR, f"metrics_plot.png"))
print(f"✅ 完成！請查看 {CURRENT_TEST_DIR} 中的 matches.jpg 連線圖與結果。")
