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

# ==========================================
# 1. 初始化與環境設定
# ==========================================
BASE_DIR = "./result_crawler"
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
GLOBAL_CSV = os.path.join(BASE_DIR, "crawler_result.csv")
HISTORY_FILE = os.path.join(BASE_DIR, "used_images.txt")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
if not os.path.exists(HISTORY_FILE):
    open(HISTORY_FILE, 'w').close()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 爬蟲工具 (Wikimedia Commons)
# ==========================================
def get_used_images():
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

def mark_image_used(filename):
    with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{filename}\n")

def fetch_images_from_wikimedia(theme_keyword, max_count):
    """透過 Wikimedia API 抓取特定主題的免費圖片 (支援無限翻頁突破版)"""
    print(f"\n🔍 正在搜尋主題：{theme_keyword} ...")
    url = "https://commons.wikimedia.org/w/api.php"

    session = requests.Session()
    honest_agent = "ImageStitchingBenchmarkBot/1.0 (Student Project; contact: academic_test@gmail.com) python-requests"
    session.headers.update({"User-Agent": honest_agent})

    used_images = get_used_images()
    downloaded_paths = []
    
    # 🌟 關鍵新增：用來儲存「下一頁號碼牌」的字典
    continue_token = {}

    # 🌟 關鍵新增：加上 while 迴圈，沒抓滿數量就一直翻頁！
    while len(downloaded_paths) < max_count:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {theme_keyword}",
            "gsrnamespace": 6,
            "gsrlimit": 50,  # 配合維基百科的單次最大極限 50
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 1024,
            "format": "json"
        }
        
        # 把下一頁的號碼牌放進參數裡 (如果是第一頁，這個 token 就是空的)
        params.update(continue_token)

        try:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"❌ API 請求失敗: {e}")
            break # 發生錯誤就先退出

        if "query" not in data or "pages" not in data["query"]:
            print(f"已經找不到更多關於 {theme_keyword} 的圖片了。")
            break

        for page_id, page_info in data["query"]["pages"].items():
            if len(downloaded_paths) >= max_count:
                break # 抓夠了就提早結束！

            if "imageinfo" in page_info:
                img_url = page_info["imageinfo"][0].get("thumburl", page_info["imageinfo"][0]["url"])
                raw_filename = urllib.parse.unquote(page_info["imageinfo"][0]["url"].split("/")[-1])
                filename = raw_filename.replace(",", "_").replace('"', '')

                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                if filename in used_images:
                    continue

                try:
                    delay = random.uniform(2.0, 4.0)
                    time.sleep(delay)

                    img_response = session.get(img_url, timeout=15)

                    if img_response.status_code == 429:
                        print(f"⚠️ 伺服器要求減速，等待 5 秒後重試...")
                        time.sleep(5.0)
                        img_response = session.get(img_url, timeout=15)

                    img_response.raise_for_status()

                    save_path = os.path.join(DOWNLOAD_DIR, filename)
                    with open(save_path, 'wb') as handler:
                        handler.write(img_response.content)

                    downloaded_paths.append((save_path, filename))
                    mark_image_used(filename)
                    print(f"✅ 下載成功 ({len(downloaded_paths)}/{max_count}): {filename}")

                except Exception as e:
                    print(f"❌ 下載失敗 {filename}: {e}")

        # 🌟 關鍵新增：檢查伺服器有沒有發「下一頁的號碼牌」給我們
        if "continue" in data:
            continue_token = data["continue"]
            print(f"👉 正在自動翻到下一頁繼續抓取...")
        else:
            print(f"庫存已見底，這個主題沒有更多圖片了！")
            break # 如果沒有號碼牌，代表真的全部抓完了

    return downloaded_paths

# ==========================================
# 3. 視覺化與核心演算法
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

def apply_random_variety(img):
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
# 4. 主流程：爬蟲 -> 測試 -> 儲存 -> 統計分析
# ==========================================
def main():
    try:
        num_images = int(input("請問每個主題要抓取幾張新照片進行測試？ "))
    except ValueError:
        print("請輸入有效的數字！")
        return

    theme_dict = {
        "farmland": "農業與植物",
        "cityscape": "城市建築",
        "indoor living room": "室內場景",
        "aerial photography": "空拍俯視",
        "mountain landscape": "自然風景"
    }

    all_results = []

    for keyword, theme_name in theme_dict.items():
        downloaded_imgs = fetch_images_from_wikimedia(keyword, num_images)

        for img_path, filename in downloaded_imgs:
            print(f"  👉 正在分析: {filename}...")

            img_orig_raw = cv2.imread(img_path)
            if img_orig_raw is None:
                continue

            h_raw, w_raw = img_orig_raw.shape[:2]
            MAX_DIM = 840
            scale_factor = MAX_DIM / max(h_raw, w_raw)
            img_orig = cv2.resize(img_orig_raw, (int(w_raw * scale_factor), int(h_raw * scale_factor)))

            file_base_name = os.path.splitext(filename)[0][:30]
            safe_name = "".join([c if c.isalnum() else "_" for c in file_base_name])
            CURRENT_TEST_DIR = os.path.join(BASE_DIR, f"{theme_name}_{safe_name}")
            os.makedirs(CURRENT_TEST_DIR, exist_ok=True)

            img_aug = apply_random_variety(img_orig)
            H_gt = get_paper_spec_h(img_orig.shape)
            h_orig, w_orig = img_orig.shape[:2]
            img_trans = cv2.warpPerspective(img_aug, H_gt, (w_orig, h_orig))

            for name in ["SIFT", "ORB", "LoFTR"]:
                H_est = match_loftr(img_orig, img_trans, CURRENT_TEST_DIR) if name == "LoFTR" else match_sift_orb(img_orig, img_trans, name, CURRENT_TEST_DIR)
                frob, cos, mse, s_val, img_wrapped = evaluate_paper_metrics(H_gt, H_est, img_aug, img_trans)

                cv2.putText(img_trans, "Transformed (GT)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
                img_wrapped_viz = img_wrapped.copy()
                cv2.putText(img_wrapped_viz, f"{name} Wrapped", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                canvas = np.hstack((img_trans, img_wrapped_viz))
                cv2.imwrite(os.path.join(CURRENT_TEST_DIR, f"{name}_vs_GT.jpg"), canvas)

                all_results.append({
                    "Theme": theme_name,
                    "Image_Name": filename,
                    "Method": name,
                    "MSE": mse,
                    "SSIM": s_val,
                    "Frobenius": frob,
                    "Cosine": cos
                })

            df_temp = pd.DataFrame(all_results[-3:])
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            metrics = [('MSE','skyblue'), ('SSIM','salmon'), ('Frobenius','green'), ('Cosine','gold')]
            for i, (m, c) in enumerate(metrics):
                df_temp.plot(x='Method', y=m, kind='bar', ax=axes[i//2, i%2], color=c, rot=0, title=m)
                if m == 'Cosine':
                    min_cos = df_temp[df_temp['Cosine'] > 0]['Cosine'].min()
                    axes[i//2, i%2].set_ylim(max(0, min_cos - 0.05), 1.005)
            plt.tight_layout()
            plt.savefig(os.path.join(CURRENT_TEST_DIR, f"metrics_plot.png"))
            plt.close()

    # ==========================================
    # 5. 終極統計分析與戰報生成 (分領域 + 總結)
    # ==========================================
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(GLOBAL_CSV, mode='a', header=not os.path.exists(GLOBAL_CSV), index=False)
        print(f"\n🎉 全部測試完成！資料已寫入 {GLOBAL_CSV}")
        
        print("\n" + "="*70)
        print(" 📊 本次測試【各領域 (Theme)】平均表現戰報")
        print("="*70)

        # 針對每個主題單獨印出戰報
        themes_tested = df['Theme'].unique()
        for theme in themes_tested:
            print(f"\n 📌 主題場景：【{theme}】")
            print("-" * 60)
            
            theme_df = df[df['Theme'] == theme]
            avg_theme_df = theme_df.groupby('Method')[['MSE', 'SSIM', 'Frobenius', 'Cosine']].mean().reset_index()
            print(avg_theme_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            
            # 找出該領域各項指標冠軍
            best_mse = avg_theme_df.loc[avg_theme_df['MSE'].idxmin()]
            best_ssim = avg_theme_df.loc[avg_theme_df['SSIM'].idxmax()]
            best_frob = avg_theme_df.loc[avg_theme_df['Frobenius'].idxmin()]
            best_cos = avg_theme_df.loc[avg_theme_df['Cosine'].idxmax()]
            
            print(f"\n 🏆 【{theme}】領域冠軍：")
            print(f"  🔸 MSE 最低:  \033[92m{best_mse['Method']}\033[0m ({best_mse['MSE']:.4f})")
            print(f"  🔸 SSIM 最高: \033[92m{best_ssim['Method']}\033[0m ({best_ssim['SSIM']:.4f})")
            print(f"  🔸 Frob. 最低:\033[92m{best_frob['Method']}\033[0m ({best_frob['Frobenius']:.4f})")
            print(f"  🔸 Cos. 最高: \033[92m{best_cos['Method']}\033[0m ({best_cos['Cosine']:.4f})")
            print("*" * 70)

        print("\n\n" + "="*70)
        print(" 🌍 本次測試【綜合總平均】跨領域決選戰報")
        print("="*70)
        
        # 依照 Method 計算全部資料的平均
        avg_df = df.groupby('Method')[['MSE', 'SSIM', 'Frobenius', 'Cosine']].mean().reset_index()
        print(avg_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("-" * 70)
        
        # 找出總冠軍
        best_overall_mse = avg_df.loc[avg_df['MSE'].idxmin()]
        best_overall_ssim = avg_df.loc[avg_df['SSIM'].idxmax()]
        best_overall_frob = avg_df.loc[avg_df['Frobenius'].idxmin()]
        best_overall_cos = avg_df.loc[avg_df['Cosine'].idxmax()]
        
        print(" 👑 各項指標【全領域綜合總冠軍】：")
        print(f"  🔸 影像復原度 (MSE 越低越好):  \033[96m{best_overall_mse['Method']}\033[0m")
        print(f"  🔸 結構相似度 (SSIM 越高越好): \033[96m{best_overall_ssim['Method']}\033[0m")
        print(f"  🔸 幾何精準度 (Frob. 越低越好): \033[96m{best_overall_frob['Method']}\033[0m")
        print(f"  🔸 矩陣相似度 (Cos. 越高越好):  \033[96m{best_overall_cos['Method']}\033[0m")
        print("="*70)

if __name__ == "__main__":
    main()
