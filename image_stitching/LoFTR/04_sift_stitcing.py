import cv2
import numpy as np
from pathlib import Path

IMG_DIR = Path("../images/20251205")
SAVE_DIR = Path("./Result/Stitched_SIFT")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
sift = cv2.SIFT_create()

def vertical_stitch_sift(plant_folder):
    all_files = list(plant_folder.glob("*.[jJ][pP][gG]")) + list(plant_folder.glob("*.[pP][nN][gG]"))
    img_up_path = next((f for f in all_files if "up" in f.name.lower()), None)
    img_down_path = next((f for f in all_files if "down" in f.name.lower()), None)
    if not img_up_path or not img_down_path: return

    img_u = cv2.imread(str(img_up_path))
    img_d = cv2.imread(str(img_down_path))

    kp1, des1 = sift.detectAndCompute(img_u, None)
    kp2, des2 = sift.detectAndCompute(img_d, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is not None:
            h_u, w_u = img_u.shape[:2]
            h_d, w_d = img_d.shape[:2]
            corners_u = np.float32([[0, 0], [0, h_u], [w_u, h_u], [w_u, 0]]).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners_u, H)
            corners_d = np.float32([[0, 0], [0, h_d], [w_d, h_d], [w_d, 0]]).reshape(-1, 1, 2)
            all_corners = np.concatenate((warped_corners, corners_d), axis=0)
            [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

            T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
            output_img = cv2.warpPerspective(img_u, T.dot(H), (x_max - x_min, y_max - y_min))
            output_img[-y_min:h_d - y_min, -x_min:w_d - x_min] = img_d

            cv2.imwrite(str(SAVE_DIR / f"{plant_folder.name}_sift_vertical.jpg"), output_img)
            print(f"SIFT 垂直縫合成功: {plant_folder.name}")

if __name__ == "__main__":
    plant_folders = sorted(list(IMG_DIR.glob("plant_*")), key=lambda x: int(x.name.split('_')[1]))
    for f in plant_folders: vertical_stitch_sift(f)
