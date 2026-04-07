import cv2
import numpy as np
import os

def build_dlt_matrix(p1, p2):
    """建立基礎的 DLT 矩陣"""
    N = len(p1)
    A = np.zeros((2 * N, 9))
    for i in range(N):
        u, v = p1[i]
        x, y = p2[i]
        A[2*i]   = [-u, -v, -1, 0, 0, 0, u*x, v*x, x]
        A[2*i+1] = [0, 0, 0, -u, -v, -1, u*y, v*y, y]
    return A

def get_local_homography(p1, p2, q, A, gamma=0.01, sigma=100.0):
    """取得單一座標點專屬的局部變形矩陣"""
    dists = np.linalg.norm(p1 - q, axis=1)
    weights = np.exp(- (dists**2) / (sigma**2))
    weights = np.maximum(weights, gamma)
    
    W = np.repeat(weights, 2)
    Aw = A * W[:, np.newaxis]
    
    _, _, V = np.linalg.svd(Aw)
    return V[-1].reshape(3, 3)

def get_canvas_bounds(h1, w1, h2, w2, H_global):
    """計算拼接後需要多大的畫布，以及第一張圖需要平移多少"""
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    corners2_warped = cv2.perspectiveTransform(corners2, H_global)
    
    all_corners = np.concatenate((np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2), corners2_warped), axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    tx, ty = -x_min, -y_min
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    
    return (x_max - x_min, y_max - y_min), T, tx, ty

def main(img1_path, img2_path):
    print("🚀 載入圖片並進行特徵匹配...")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("❌ 找不到圖片！") 
        return
        
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 這裡使用 SIFT 作為示範，你可以把這裡替換成你 LoFTR 找出來的 pts1 和 pts2
    sift = cv2.SIFT_create(2000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = matcher.knnMatch(des2, des1, k=2) # 注意：是從 img2 匹配到 img1
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    pts2 = np.float32([kp2[m.queryIdx].pt for m in good_matches])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches])
    
    H_global, inlier_mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    inliers2 = pts2[inlier_mask.ravel() == 1]
    inliers1 = pts1[inlier_mask.ravel() == 1]

    # 計算大畫布尺寸
    canvas_size, T, tx, ty = get_canvas_bounds(h1, w1, h2, w2, H_global)
    W_canvas, H_canvas = canvas_size

    # ==========================================
    # 1. 傳統 Global Homography 拼接與紅框
    # ==========================================
    print("📸 正在生成 Global Homography 拼接圖...")
    canvas_global = cv2.warpPerspective(img2, T @ H_global, (W_canvas, H_canvas))
    canvas_global[ty:ty+h1, tx:tx+w1] = img1 # 把左圖疊加上去
    
    # 畫紅框 (代表單一矩陣)
    corners_homo = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    corners_homo_warped = cv2.perspectiveTransform(corners_homo, T @ H_global)
    cv2.polylines(canvas_global, [np.int32(corners_homo_warped)], True, (0, 0, 255), 5)
    cv2.putText(canvas_global, "Global Homography (Rigid Stitching)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # ==========================================
    # 2. APAP (Moving DLT) 拼接與綠色網格
    # ==========================================
    print("🔥 正在進行 APAP 像素級局部縫合 (請稍候幾秒鐘)...")
    canvas_apap = np.zeros((H_canvas, W_canvas, 3), dtype=np.uint8)
    canvas_apap[ty:ty+h1, tx:tx+w1] = img1 # 把左圖疊加上去
    
    cols, rows = 40, 30
    X = np.linspace(0, w2, cols)
    Y = np.linspace(0, h2, rows)
    A = build_dlt_matrix(inliers2, inliers1)
    sigma_val = max(w2, h2) * 0.1

    # 【步驟 A】切片扭曲：把照片切成一塊塊貼上去
    for r in range(rows - 1):
        for c in range(cols - 1):
            # 取網格中心點算 H_local
            cx, cy = (X[c] + X[c+1])/2, (Y[r] + Y[r+1])/2
            H_local = get_local_homography(inliers2, inliers1, np.array([[cx, cy]]), A, sigma=sigma_val)
            H_final = T @ H_local
            
            # 定義這一小塊的四個頂點
            cell_corners = np.float32([[X[c], Y[r]], [X[c+1], Y[r]], [X[c+1], Y[r+1]], [X[c], Y[r+1]]])
            cell_warped = cv2.perspectiveTransform(cell_corners.reshape(-1, 1, 2), H_final).reshape(-1, 2)
            
            # 取得這塊小拼圖在畫布上的範圍
            xmin, ymin = np.int32(cell_warped.min(axis=0))
            xmax, ymax = np.int32(cell_warped.max(axis=0))
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(W_canvas, xmax), min(H_canvas, ymax)
            
            if xmax > xmin and ymax > ymin:
                # 局部加速扭曲
                T_patch = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
                patch_img = cv2.warpPerspective(img2, T_patch @ H_final, (xmax - xmin, ymax - ymin))
                
                # 建立遮罩並貼上畫布
                mask = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.int32(cell_warped - [xmin, ymin]), 255)
                roi = canvas_apap[ymin:ymax, xmin:xmax]
                np.copyto(roi, patch_img, where=(mask == 255)[..., None])

    # 【步驟 B】畫上具有彈性的綠色網格
    grid_warped_pts = np.zeros((rows, cols, 2))
    for r in range(rows):
        for c in range(cols):
            q = np.array([[X[c], Y[r]]])
            H_local = get_local_homography(inliers2, inliers1, q, A, sigma=sigma_val)
            pt_warped = cv2.perspectiveTransform(q.reshape(-1, 1, 2), T @ H_local)
            grid_warped_pts[r, c] = pt_warped.ravel()
            
    # 畫橫線與直線
    for r in range(rows):
        for c in range(cols - 1):
            cv2.line(canvas_apap, tuple(np.int32(grid_warped_pts[r, c])), tuple(np.int32(grid_warped_pts[r, c+1])), (0, 255, 0), 2)
    for c in range(cols):
        for r in range(rows - 1):
            cv2.line(canvas_apap, tuple(np.int32(grid_warped_pts[r, c])), tuple(np.int32(grid_warped_pts[r+1, c])), (0, 255, 0), 2)

    cv2.putText(canvas_apap, "APAP (Local Mesh Stitching)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # 儲存圖片
    output_dir = "./stitch_comparison"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "01_Global_Stitch.jpg"), canvas_global)
    cv2.imwrite(os.path.join(output_dir, "02_APAP_Stitch.jpg"), canvas_apap)
    print("🎉 完美縫合完成！請前往 `./stitch_comparison` 查看結果！")

if __name__ == "__main__":
    img1_path = "40.jpg"  # 左圖
    img2_path = "70.jpg"  # 右圖 (網格會畫在這張上)
    main(img1_path, img2_path)
