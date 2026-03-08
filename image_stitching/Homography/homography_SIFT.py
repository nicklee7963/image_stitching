import cv2
import numpy as np
import os
import glob

# 設定路徑
BASE_IMAGE_DIR = "../images/20251205"
RESULT_DIR = "Result_Raw"

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def stitch_raw(up_path, down_path, plant_name):
    img_up = cv2.imread(up_path)
    img_down = cv2.imread(down_path)
    
    if img_up is None or img_down is None:
        return
        
    gray_up = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    gray_down = cv2.cvtColor(img_down, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_up, des_up = sift.detectAndCompute(gray_up, None)
    kp_down, des_down = sift.detectAndCompute(gray_down, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    if des_up is None or des_down is None:
        return
        
    matches = flann.knnMatch(des_up, des_down, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        print(f"[{plant_name}] 匹配點不足，無法計算矩陣。")
        return

    src_pts = np.float32([kp_up[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_down[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print(f"[{plant_name}] H 矩陣計算失敗。")
        return

    h_up, w_up = img_up.shape[:2]
    h_down, w_down = img_down.shape[:2]

    # 計算 Up.jpg 變形後的位置
    corners_up = np.float32([[0, 0], [0, h_up], [w_up, h_up], [w_up, 0]]).reshape(-1, 1, 2)
    warped_corners_up = cv2.perspectiveTransform(corners_up, H)
    
    # 加入 Down.jpg 的位置以決定總畫布大小
    corners_down = np.float32([[0, 0], [0, h_down], [w_down, h_down], [w_down, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((warped_corners_up, corners_down), axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    
    # 安全機制：如果特徵錯配導致矩陣將圖片投影到無限遠，記憶體會直接崩潰。這裡設定一個合理的畫布上限。
    if canvas_w > 15000 or canvas_h > 15000:
        print(f"[{plant_name}] 矩陣崩潰，算出的畫布過大 ({canvas_w}x{canvas_h})，已攔截以防當機。")
        return

    # 執行 Up.jpg 的變形
    canvas = cv2.warpPerspective(img_up, H_translation.dot(H), (canvas_w, canvas_h))
    
    # 畫上紅色粗邊框，標示出 Up.jpg 變形後的輪廓
    warped_corners_up_translated = warped_corners_up + translation_dist
    cv2.polylines(canvas, [np.int32(warped_corners_up_translated)], True, (0, 0, 255), 15)

    # 暴力覆蓋 Down.jpg
    down_y_start = translation_dist[1]
    down_x_start = translation_dist[0]
    canvas[down_y_start:down_y_start+h_down, down_x_start:down_x_start+w_down] = img_down

    result_path = os.path.join(RESULT_DIR, f"{plant_name}_raw_stitch.jpg")
    cv2.imwrite(result_path, canvas)
    print(f"[{plant_name}] 原始矩陣疊加完成，儲存至 {result_path}")

def main():
    plant_folders = glob.glob(os.path.join(BASE_IMAGE_DIR, "plant_*"))
    for folder in plant_folders:
        plant_name = os.path.basename(folder)
        stitch_raw(os.path.join(folder, "Up.jpg"), os.path.join(folder, "Down.jpg"), plant_name)

if __name__ == "__main__":
    main()
