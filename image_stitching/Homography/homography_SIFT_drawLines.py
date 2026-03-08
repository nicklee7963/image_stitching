import cv2
import os

# 設定路徑 (直接鎖定案發現場 plant_1)
up_path = "../images/20251205/plant_1/Up.jpg"
down_path = "../images/20251205/plant_1/Down.jpg"
output_path = "Result/plant_1_sift_matches.jpg"

def main():
    # 確保輸出資料夾存在
    if not os.path.exists("Result"):
        os.makedirs("Result")

    # 讀取圖片
    img_up = cv2.imread(up_path)
    img_down = cv2.imread(down_path)

    if img_up is None or img_down is None:
        print("❌ 圖片讀取失敗，請確認路徑是否正確。")
        return

    # 轉灰階
    gray_up = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    gray_down = cv2.cvtColor(img_down, cv2.COLOR_BGR2GRAY)

    # 1. 建立 SIFT 提取器
    sift = cv2.SIFT_create()
    kp_up, des_up = sift.detectAndCompute(gray_up, None)
    kp_down, des_down = sift.detectAndCompute(gray_down, None)

    # 2. 特徵匹配 (使用 FLANN)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_up, des_down, k=2)

    # Lowe's ratio test 過濾優良匹配點
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 3. 畫出匹配連線
    # matchColor=(0, 255, 0) 設定連線為綠色
    # flags=2 表示不畫出沒有匹配成功的孤立特徵點，畫面比較乾淨
    matched_img = cv2.drawMatches(img_up, kp_up, img_down, kp_down, good_matches, None,
                                  matchColor=(0, 255, 0), 
                                  singlePointColor=(255, 0, 0), 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 儲存結果
    cv2.imwrite(output_path, matched_img)
    print(f"✅ SIFT 匹配連線圖已儲存至: {output_path}")
    print(f"🔍 總共找到 {len(good_matches)} 個初步匹配點 (請打開圖片檢查連線是否混亂或交叉)。")

if __name__ == "__main__":
    main()
