import time
import random
import urllib.parse
import requests
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torch
import kornia as K
import kornia.feature as KF
import re

# ==========================================
# 1. 初始化與環境設定
# ==========================================
BASE_DIR = "./result_crawler"
GLOBAL_CSV = os.path.join(BASE_DIR, "crawler_result.csv")
HISTORY_FILE = os.path.join(BASE_DIR, "used_images.txt")

# 🌟 建立本次執行的最高層級資料夾 (如: 20260315_1617)
RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M")
RUN_DIR = os.path.join(BASE_DIR, RUN_TIMESTAMP)
# 🌟 建立 result 資料夾
RESULT_DIR = os.path.join(RUN_DIR, "result")

os.makedirs(RESULT_DIR, exist_ok=True)
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR, exist_ok=True)
# if not os.path.exists(HISTORY_FILE):
#    open(HISTORY_FILE, 'w').close()
open(HISTORY_FILE, 'w').close()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 互動式參數設定工具
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
    print(" ⚙️ 實驗難度參數設定 (挑戰 SIFT 的極限！)")
    print("="*50)
    config = {}
    config['blur_prob'] = get_valid_input("🔸 [1] 模糊發生機率", 0.3, 0.0, 1.0)
    config['alpha_min'] = get_valid_input("🔸 [2] 對比度變化下限", 0.6, 0.1, 1.0)
    config['alpha_max'] = get_valid_input("🔸 [3] 對比度變化上限", 1.4, 1.0, 3.0)
    config['beta_max'] = get_valid_input("🔸 [4] 亮度最大偏移量", 40, 0, 100, is_int=True)
    config['angle_max'] = get_valid_input("🔸 [5] 最大旋轉角度", 40, 0, 180)
    print("="*50 + "\n")
    return config

# ==========================================
# 3. 爬蟲工具 (Wikimedia Commons)
# ==========================================
def get_used_images():
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

def mark_image_used(filename):
    with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{filename}\n")

def fetch_images_from_wikimedia(theme_keyword, theme_name, max_count):
    print(f"\n🔍 正在搜尋主題：{theme_keyword} ...")
    url = "https://commons.wikimedia.org/w/api.php"

    session = requests.Session()
    honest_agent = "ImageStitchingBenchmarkBot/1.0 (Student Project; contact: academic_test@gmail.com) python-requests"
    session.headers.update({"User-Agent": honest_agent})

    used_images = get_used_images()
    downloaded_paths = []
    continue_token = {}

    theme_dir = os.path.join(RESULT_DIR, theme_name)
    os.makedirs(theme_dir, exist_ok=True)

    while len(downloaded_paths) < max_count:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {theme_keyword}",
            "gsrnamespace": 6,
            "gsrlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 1024,
            "format": "json"
        }
        params.update(continue_token)

        try:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"❌ API 請求失敗: {e}")
            break

        if "query" not in data or "pages" not in data["query"]:
            print(f"已經找不到更多關於 {theme_keyword} 的圖片了。")
            break

        for page_id, page_info in data["query"]["pages"].items():
            if len(downloaded_paths) >= max_count:
                break

            if "imageinfo" in page_info:
                img_url = page_info["imageinfo"][0].get("thumburl", page_info["imageinfo"][0]["url"])
                raw_filename = urllib.parse.unquote(page_info["imageinfo"][0]["url"].split("/")[-1])
                filename = raw_filename.replace(",", "_").replace('"', '')

                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                if filename in used_images:
                    continue

                try:
                    delay = random.uniform(1.0, 3.0)
                    time.sleep(delay)

                    img_response = session.get(img_url, timeout=15)
                    if img_response.status_code == 429:
                        print(f"⚠️ 伺服器要求減速，等待 5 秒後重試...")
                        time.sleep(5.0)
                        img_response = session.get(img_url, timeout=15)
                    img_response.raise_for_status()

                    safe_name = "".join([c if c.isalnum() else "_" for c in filename.split('.')[0]][:30])
                    img_specific_dir = os.path.join(theme_dir, safe_name)
                    os.makedirs(img_specific_dir, exist_ok=True)

                    save_path = os.path.join(img_specific_dir, filename)
                    with open(save_path, 'wb') as handler:
                        handler.write(img_response.content)

                    downloaded_paths.append((save_path, filename, img_specific_dir, safe_name))
                    mark_image_used(filename)
                    print(f"✅ 下載成功 ({len(downloaded_paths)}/{max_count}): {filename}")

                except Exception as e:
                    print(f"❌ 下載失敗 {filename}: {e}")

        if "continue" in data:
            continue_token = data["continue"]
            print(f"👉 正在自動翻到下一頁繼續抓取...")
        else:
            print(f"庫存已見底，這個主題沒有更多圖片了！")
            break

    return downloaded_paths

# ==========================================
# 4. 視覺化與核心演算法
# ==========================================
def draw_matches(img1, img2, pts1, pts2, name, test_dir):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2
    for p1, p2 in zip(pts1[:80], pts2[:80]):
        pt1 = (int(p1[0][0]), int(p1[0][1]))
        pt2 = (int(p2[0][0]) + w1, int(p2[0][1]))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(canvas, pt1, 5, color, -1)
        cv2.circle(canvas, pt2, 5, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1)
    cv2.imwrite(os.path.join(test_dir, f"{name}_matches.jpg"), canvas)

def apply_random_variety(img, config):
    alpha = np.random.uniform(config['alpha_min'], config['alpha_max'])
    beta = np.random.randint(-config['beta_max'], config['beta_max'] + 1) if config['beta_max'] > 0 else 0
    img_aug = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if np.random.rand() < config['blur_prob']:
        img_aug = cv2.GaussianBlur(img_aug, (11, 11), 0)
    h, w = img_aug.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    cx, cy = np.random.randint(0, w), np.random.randint(0, h)
    cv2.circle(mask, (cx, cy), np.random.randint(150, 250), 0.3, -1)
    mask = cv2.GaussianBlur(mask, (101, 101), 0)
    return (img_aug * mask[:, :, np.newaxis]).astype(np.uint8)

def get_paper_spec_h(shape, config):
    h, w = shape[:2]
    cx, cy = w / 2, h / 2
    angle = np.random.uniform(-config['angle_max'], config['angle_max'])
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

def match_sift_orb(img_clean, img_dirty, mode="SIFT", test_dir=None):
    det = cv2.SIFT_create() if mode == "SIFT" else cv2.ORB_create(3000)
    gray2 = cv2.cvtColor(img_dirty, cv2.COLOR_BGR2GRAY)
    _, mask2 = cv2.threshold(gray2, 1, 255, cv2.THRESH_BINARY)
    mask2 = cv2.erode(mask2, np.ones((15, 15), np.uint8), iterations=2)
    kp1, des1 = det.detectAndCompute(img_clean, None)
    kp2, des2 = det.detectAndCompute(img_dirty, mask2)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4: return np.eye(3)
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
    pts1_raw, pts2_raw = out['keypoints0'].cpu().numpy(), out['keypoints1'].cpu().numpy()
    conf = out['confidence'].cpu().numpy()
    sort_idx = np.argsort(conf)[::-1]
    pts1_raw, pts2_raw = pts1_raw[sort_idx], pts2_raw[sort_idx]
    pts1, pts2 = np.zeros_like(pts1_raw), np.zeros_like(pts2_raw)
    pts1[:, 0], pts1[:, 1] = pts1_raw[:, 0] * (w / new_w), pts1_raw[:, 1] * (h / new_h)
    pts2[:, 0], pts2[:, 1] = pts2_raw[:, 0] * (w / new_w), pts2_raw[:, 1] * (h / new_h)
    if len(pts1) > 10:
        pts1_reshaped = np.float32(pts1).reshape(-1, 1, 2)
        pts2_reshaped = np.float32(pts2).reshape(-1, 1, 2)
        if test_dir: draw_matches(img_clean, img_dirty, pts1_reshaped, pts2_reshaped, "LoFTR", test_dir)
        H_est, _ = cv2.findHomography(pts1_reshaped, pts2_reshaped, cv2.RANSAC, 4.0)
        return H_est if H_est is not None else np.eye(3)
    return np.eye(3)

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
# 5. 主流程：爬蟲 -> 測試 -> 儲存 -> 統計分析
# ==========================================
def main():
    try:
        num_images = int(input("請問每個主題要抓取幾張新照片進行測試？ "))
    except ValueError:
        print("請輸入有效的數字！")
        return

    exp_config = setup_experiment_config()

    # 🌟 第一時間儲存實驗參數的 txt 紀錄
    config_path = os.path.join(RESULT_DIR, "experiment_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(" ⚙️ 實驗參數設定紀錄 (Experiment Configuration)\n")
        f.write("="*50 + "\n")
        f.write(f"🔸 模糊發生機率 (Blur Probability) : {exp_config['blur_prob']}\n")
        f.write(f"🔸 對比度變化下限 (Alpha Min)      : {exp_config['alpha_min']}\n")
        f.write(f"🔸 對比度變化上限 (Alpha Max)      : {exp_config['alpha_max']}\n")
        f.write(f"🔸 亮度最大偏移量 (Beta Max)       : {exp_config['beta_max']}\n")
        f.write(f"🔸 最大旋轉角度 (Max Angle)        : {exp_config['angle_max']}\n")
        f.write("="*50 + "\n")

    theme_dict = {
        "farmland": "農業與植物",
        "cityscape": "城市建築",
        "indoor living room": "室內場景",
        "aerial photography": "空拍俯視",
        "mountain landscape": "自然風景"
    }

    all_results = []

    for keyword, theme_name in theme_dict.items():
        downloaded_imgs = fetch_images_from_wikimedia(keyword, theme_name, num_images)

        for img_path, filename, img_specific_dir, safe_name in downloaded_imgs:
            print(f"  👉 正在分析: {filename}...")

            img_orig_raw = cv2.imread(img_path)
            if img_orig_raw is None:
                continue

            h_raw, w_raw = img_orig_raw.shape[:2]
            MAX_DIM = 840
            scale_factor = MAX_DIM / max(h_raw, w_raw)
            img_orig = cv2.resize(img_orig_raw, (int(w_raw * scale_factor), int(h_raw * scale_factor)))

            img_aug = apply_random_variety(img_orig, exp_config)
            H_gt = get_paper_spec_h(img_orig.shape, exp_config)
            
            h_orig, w_orig = img_orig.shape[:2]
            img_trans = cv2.warpPerspective(img_aug, H_gt, (w_orig, h_orig))

            single_img_results = []

            for name in ["SIFT", "ORB", "LoFTR"]:
                H_est = match_loftr(img_orig, img_trans, img_specific_dir) if name == "LoFTR" else match_sift_orb(img_orig, img_trans, name, img_specific_dir)
                frob, cos, mse, s_val, img_wrapped = evaluate_paper_metrics(H_gt, H_est, img_aug, img_trans)

                cv2.putText(img_trans, "Transformed (GT)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
                img_wrapped_viz = img_wrapped.copy()
                cv2.putText(img_wrapped_viz, f"{name} Wrapped", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                canvas = np.hstack((img_trans, img_wrapped_viz))
                cv2.imwrite(os.path.join(img_specific_dir, f"{name}_vs_GT.jpg"), canvas)

                res_dict = {
                    "Theme": theme_name,
                    "Image_Name": filename,
                    "Method": name,
                    "MSE": mse,
                    "SSIM": s_val,
                    "Frobenius": frob,
                    "Cosine": cos
                }
                single_img_results.append(res_dict)
                all_results.append(res_dict)

            df_single = pd.DataFrame(single_img_results)
            df_single.to_csv(os.path.join(img_specific_dir, f"{safe_name}.csv"), index=False, encoding='utf-8-sig')

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            metrics = [('MSE','skyblue'), ('SSIM','salmon'), ('Frobenius','green'), ('Cosine','gold')]
            for i, (m, c) in enumerate(metrics):
                df_single.plot(x='Method', y=m, kind='bar', ax=axes[i//2, i%2], color=c, rot=0, title=m)
                if m == 'Cosine':
                    min_cos = df_single[df_single['Cosine'] > 0]['Cosine'].min()
                    axes[i//2, i%2].set_ylim(max(0, min_cos - 0.05), 1.005)
            plt.tight_layout()
            plt.savefig(os.path.join(img_specific_dir, f"metrics_plot.png"))
            plt.close()

    # ==========================================
    # 6. 終極統計分析與戰報生成
    # ==========================================
    if all_results:
        df = pd.DataFrame(all_results)
        
        df.to_csv(GLOBAL_CSV, mode='a', header=not os.path.exists(GLOBAL_CSV), index=False, encoding='utf-8-sig')
        CURRENT_RUN_CSV = os.path.join(RESULT_DIR, f"run_summary.csv")
        df.to_csv(CURRENT_RUN_CSV, index=False, encoding='utf-8-sig')
        
        report_lines = []
        def log(text):
            print(text)
            clean_text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
            report_lines.append(clean_text)

        log("\n" + "="*70)
        log(" ⚙️ 實驗參數設定回顧")
        log("="*70)
        log(f"🔸 模糊發生機率: {exp_config['blur_prob']} | 對比度變化下限: {exp_config['alpha_min']} | 上限: {exp_config['alpha_max']}")
        log(f"🔸 亮度最大偏移: {exp_config['beta_max']} | 最大旋轉角度: {exp_config['angle_max']}")
        
        log("\n" + "="*70)
        log(f" 📂 儲存路徑總覽 (潔癖收納版)")
        log("="*70)
        log(f"  🔹 歷史總和庫 CSV：   \033[96m{GLOBAL_CSV}\033[0m")
        log(f"  🔹 本次詳細結果包：   \033[96m{RUN_DIR}/\033[0m")
        log("="*70)
        
        log("\n 📊 本次測試【各領域 (Theme)】平均表現戰報")
        log("="*70)

        themes_tested = df['Theme'].unique()
        for theme in themes_tested:
            log(f"\n 📌 主題場景：【{theme}】")
            log("-" * 60)
            
            theme_df = df[df['Theme'] == theme]
            avg_theme_df = theme_df.groupby('Method')[['MSE', 'SSIM', 'Frobenius', 'Cosine']].mean().reset_index()
            log(avg_theme_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            
            best_mse = avg_theme_df.loc[avg_theme_df['MSE'].idxmin()]
            best_ssim = avg_theme_df.loc[avg_theme_df['SSIM'].idxmax()]
            best_frob = avg_theme_df.loc[avg_theme_df['Frobenius'].idxmin()]
            best_cos = avg_theme_df.loc[avg_theme_df['Cosine'].idxmax()]
            
            log(f"\n 🏆 【{theme}】領域冠軍：")
            log(f"  🔸 MSE 最低:  \033[92m{best_mse['Method']}\033[0m ({best_mse['MSE']:.4f})")
            log(f"  🔸 SSIM 最高: \033[92m{best_ssim['Method']}\033[0m ({best_ssim['SSIM']:.4f})")
            log(f"  🔸 Frob. 最低:\033[92m{best_frob['Method']}\033[0m ({best_frob['Frobenius']:.4f})")
            log(f"  🔸 Cos. 最高: \033[92m{best_cos['Method']}\033[0m ({best_cos['Cosine']:.4f})")
            log("*" * 70)

        log("\n\n" + "="*70)
        log(" 🌍 本次測試【綜合總平均】跨領域決選戰報")
        log("="*70)
        
        avg_df = df.groupby('Method')[['MSE', 'SSIM', 'Frobenius', 'Cosine']].mean().reset_index()
        log(avg_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        log("-" * 70)
        
        best_overall_mse = avg_df.loc[avg_df['MSE'].idxmin()]
        best_overall_ssim = avg_df.loc[avg_df['SSIM'].idxmax()]
        best_overall_frob = avg_df.loc[avg_df['Frobenius'].idxmin()]
        best_overall_cos = avg_df.loc[avg_df['Cosine'].idxmax()]
        
        log(" 👑 各項指標【全領域綜合總冠軍】：")
        log(f"  🔸 影像復原度 (MSE 越低越好):  \033[96m{best_overall_mse['Method']}\033[0m")
        log(f"  🔸 結構相似度 (SSIM 越高越好): \033[96m{best_overall_ssim['Method']}\033[0m")
        log(f"  🔸 幾何精準度 (Frob. 越低越好): \033[96m{best_overall_frob['Method']}\033[0m")
        log(f"  🔸 矩陣相似度 (Cos. 越高越好):  \033[96m{best_overall_cos['Method']}\033[0m")
        log("="*70)

        report_path = os.path.join(RESULT_DIR, "battle_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

if __name__ == "__main__":
    main()
