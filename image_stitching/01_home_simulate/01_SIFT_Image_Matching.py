import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

# Set global plot size
plt.rcParams['figure.figsize'] = [15, 15]

# 1. Read image and convert them to gray
def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot find image at {path}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb

# 2. SIFT Feature Detection
def SIFT(img_gray):
    # Depending on OpenCV version
    try:
        siftDetector = cv2.SIFT_create()
    except AttributeError:
        siftDetector = cv2.xfeatures2d.SIFT_create()
        
    kp, des = siftDetector.detectAndCompute(img_gray, None)
    return kp, des

def plot_sift(gray, rgb, kp):
    tmp = rgb.copy()
    img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

# 3. Keypoint Matching
# Fixed the argument count to match your call
def matcher(kp1, des1, img1, kp2, des2, img2, threshold):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])

    matches_pts = []
    for pair in good:
        # Extract the (x, y) coordinates for both images
        matches_pts.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    return np.array(matches_pts)

# 4. Homography & RANSAC Logic
def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H / H[2, 2] 
    return H

def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx]
    return np.array(point)

def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp / temp[2])[0:2]
    errors = np.linalg.norm(all_p2 - estimate_p2, axis=1) ** 2
    return errors

def ransac(matches, threshold, iters):
    num_best_inliers = 0
    best_H = np.eye(3)
    best_inliers = []
    
    for i in range(iters):
        points = random_point(matches)
        H = homography(points)
        
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print(f"inliers/matches: {num_best_inliers}/{len(matches)}")
    return best_inliers, best_H

# 5. Image Stitching
def stitch_img(left, right, H):
    print("Stitching image... (this uses manual loops and will be slow)")
    
    # Normalize to float 0-1
    left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   
    right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   
    
    # left image geometry
    height_l, width_l, _ = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min, x_min = min(y_news), min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H_translated = np.dot(translation_mat, H)
    
    # Canvas Size
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # Warp
    warped_l = cv2.warpPerspective(src=left, M=H_translated, dsize=size)
    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
     
    black = np.zeros(3)
    
    # Manual Stitching Loop (Exact logic from the blog)
    # Using range(warped_l.shape...) ensures we cover the whole canvas
    for i in tqdm(range(warped_l.shape[0])):
        for j in range(warped_l.shape[1]):
            pixel_l = warped_l[i, j, :]
            # Check bounds for warped_r as it might be smaller or translated
            if i < warped_r.shape[0] and j < warped_r.shape[1]:
                pixel_r = warped_r[i, j, :]
            else:
                pixel_r = black
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                # Ghosting happens here
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
                  
    return warped_l

# --- Main Logic ---

# 1. Load
left_gray, _, left_rgb = read_image('image/tv_desk/40.jpg')
right_gray, _, right_rgb = read_image('image/tv_desk/70.jpg')

# 2. SIFT
kp_l, des_l = SIFT(left_gray)
kp_r, des_r = SIFT(right_gray)

# 3. Match
matches = matcher(kp_l, des_l, left_rgb, kp_r, des_r, right_rgb, 0.5)

# 4. RANSAC
inliers, H_matrix = ransac(matches, 0.5, 2000)

# 5. Stitch & Show
final_result = stitch_img(left_rgb, right_rgb, H_matrix)

plt.imshow(final_result)
plt.axis('off')
plt.title("Neutrino Blog Implementation Result")
plt.show()
