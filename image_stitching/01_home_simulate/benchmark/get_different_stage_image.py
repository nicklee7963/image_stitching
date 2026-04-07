import cv2
import numpy as np
import os

# ==========================================
# 1. 互動式參數設定工具 (包含所有幾何參數)
# ==========================================
def get_valid_input(prompt, default, min_val, max_val, is_int=False):
    while True:
        val = input(f"{prompt} (可接受範圍: {min_val} ~ {max_val}) [按 Enter 使用預設 {default}]: ")
        if not val.strip():
            return default
        try:
            val = int(val) if is_int else float(val)
            if min_val <= val <= max_val:
                return val
            else:
                print(f"⚠️ 數值超出範圍！請輸入 {min_val} 到 {max_val} 之間的數字。")
        except ValueError:
            print("⚠️ 格式錯誤！請輸入數字。")

def setup_experiment_config():
    print("\n" + "="*50)
    print(" ⚙️ 簡報展示：干擾與變形參數設定")
    print("="*50)
    config = {}
    print("【光學干擾設定 Photometric】")
    config['blur_prob'] = 1.0 # 為了展示，強制一定發生模糊
    config['alpha_val'] = get_valid_input("🔸 對比度變化 (預設 0.6 變灰暗)", 0.6, 0.1, 3.0)
    config['beta_val'] = get_valid_input("🔸 亮度偏移量 (預設 40 變亮)", 40, -100, 100, is_int=True)
    
    print("\n【幾何變形設定 Geometric (Homography)】")
    config['angle_val'] = get_valid_input("🔸 旋轉角度 (度)", 30, -180, 180)
    config['scale_val'] = get_valid_input("🔸 縮放比例 (預設 0.8 變小)", 0.8, 0.1, 3.0)
    config['trans_ratio'] = get_valid_input("🔸 平移比例 (預設 0.1 代表偏移 10%)", 0.1, 0.0, 0.5)
    config['shear_val'] = get_valid_input("🔸 錯切形變 (Shear)", 0.1, 0.0, 0.5)
    config['persp_val'] = get_valid_input("🔸 透視形變 (Perspective)", 0.0005, 0.0, 0.002)
    print("="*50 + "\n")
    return config

# ==========================================
# 2. 建立儲存資料夾與讀取圖片
# ==========================================
# 🌟 請在這裡填入你要用來當 Demo 的照片路徑
IMG_PATH = "image.png" 
OUTPUT_DIR = "./presentation_demo_images"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

img_orig = cv2.imread(IMG_PATH)
if img_orig is None:
    print(f"❌ 找不到測試圖片: {IMG_PATH}")
    exit()

h, w = img_orig.shape[:2]
# 縮小一點方便儲存與檢視
scale_factor = 840 / max(h, w)
img_orig = cv2.resize(img_orig, (int(w * scale_factor), int(h * scale_factor)))
h, w = img_orig.shape[:2]

config = setup_experiment_config()

print(f"\n🚀 正在產出簡報用的拆解步驟圖至資料夾: {OUTPUT_DIR} ...")

# 存下第一張：原始圖片
cv2.imwrite(os.path.join(OUTPUT_DIR, "00_Original_Image.jpg"), img_orig)

# ==========================================
# 3. 光學干擾拆解 (Photometric)
# ==========================================
# 1. 單純改變亮度 (Brightness)
img_bright = cv2.convertScaleAbs(img_orig, alpha=1.0, beta=config['beta_val'])
cv2.imwrite(os.path.join(OUTPUT_DIR, "01_Brightness_Only.jpg"), img_bright)

# 2. 單純改變對比度 (Contrast)
img_contrast = cv2.convertScaleAbs(img_orig, alpha=config['alpha_val'], beta=0)
cv2.imwrite(os.path.join(OUTPUT_DIR, "02_Contrast_Only.jpg"), img_contrast)

# 3. 單純模糊 (Blur)
img_blur = cv2.GaussianBlur(img_orig, (15, 15), 0)
cv2.imwrite(os.path.join(OUTPUT_DIR, "03_Blur_Only.jpg"), img_blur)

# 4. 單純加上局部陰影 (Shadow)
mask = np.ones((h, w), dtype=np.float32)
cx, cy = int(w*0.6), int(h*0.4) # 固定在畫面偏右上方
cv2.circle(mask, (cx, cy), 200, 0.3, -1)
mask = cv2.GaussianBlur(mask, (101, 101), 0)
img_shadow = (img_orig * mask[:, :, np.newaxis]).astype(np.uint8)
cv2.imwrite(os.path.join(OUTPUT_DIR, "04_Shadow_Only.jpg"), img_shadow)

# 5. 光學干擾全餐 (Augmented Image)
img_aug_all = cv2.convertScaleAbs(img_orig, alpha=config['alpha_val'], beta=config['beta_val'])
img_aug_all = cv2.GaussianBlur(img_aug_all, (11, 11), 0)
img_aug_all = (img_aug_all * mask[:, :, np.newaxis]).astype(np.uint8)
cv2.imwrite(os.path.join(OUTPUT_DIR, "05_All_Photometric_Combined.jpg"), img_aug_all)

# ==========================================
# 4. 幾何變形拆解 (Geometric Homography)
# 為了讓幾何變形看得更清楚，我們拿「原始圖片」來做獨立扭曲
# ==========================================
cx, cy = w / 2, h / 2

# 1. 單純縮放 (Scale)
M_scale = cv2.getRotationMatrix2D((cx, cy), 0, config['scale_val'])
H_scale = np.vstack([M_scale, [0, 0, 1]])
img_scale = cv2.warpPerspective(img_orig, H_scale, (w, h))
cv2.imwrite(os.path.join(OUTPUT_DIR, "06_Scale_Only.jpg"), img_scale)

# 2. 單純平移 (Translation)
tx, ty = w * config['trans_ratio'], h * config['trans_ratio']
H_trans = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
img_trans = cv2.warpPerspective(img_orig, H_trans, (w, h))
cv2.imwrite(os.path.join(OUTPUT_DIR, "07_Translation_Only.jpg"), img_trans)

# 3. 單純錯切 (Shear)
shx, shy = config['shear_val'], config['shear_val']
H_shear = np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]], dtype=np.float32)
img_shear = cv2.warpPerspective(img_orig, H_shear, (w, h))
cv2.imwrite(os.path.join(OUTPUT_DIR, "08_Shear_Only.jpg"), img_shear)

# 4. 單純透視 (Perspective)
px, py = config['persp_val'], config['persp_val']
H_persp = np.array([[1, 0, 0], [0, 1, 0], [px, py, 1]], dtype=np.float32)
img_persp = cv2.warpPerspective(img_orig, H_persp, (w, h))
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_Perspective_Only.jpg"), img_persp)

# 5. 單純旋轉 (Rotation)
M_rot = cv2.getRotationMatrix2D((cx, cy), config['angle_val'], 1.0)
H_rot = np.vstack([M_rot, [0, 0, 1]])
img_rot = cv2.warpPerspective(img_orig, H_rot, (w, h))
cv2.imwrite(os.path.join(OUTPUT_DIR, "10_Rotation_Only.jpg"), img_rot)

# ==========================================
# 5. 終極完全體 (最終丟給演算法考試的圖)
# 將所有幾何變形加總，並套用在「光學干擾全餐圖」上
# ==========================================
H_final = H_persp @ H_trans @ H_shear @ H_rot @ H_scale
H_final = H_final / H_final[2, 2]

img_ultimate_transformed = cv2.warpPerspective(img_aug_all, H_final, (w, h))
cv2.imwrite(os.path.join(OUTPUT_DIR, "11_Ultimate_Transformed_Final.jpg"), img_ultimate_transformed)

print("🎉 全部完成！請去 `./presentation_demo_images` 資料夾看看你的簡報素材！")
