import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['figure.figsize'] = [15, 10]

# 1. Image Loading & Resizing
def read_and_resize(path, max_dim=800):
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img_rgb

left_gray, left_rgb = read_and_resize('image/tv_desk/40.jpg')
right_gray, right_rgb = read_and_resize('image/tv_desk/45.jpg')

# 2. SIFT & Aggressive Matching
try:
    sift = cv2.SIFT_create(nfeatures=5000)
except AttributeError:
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=5000)

kp_l, des_l = sift.detectAndCompute(left_gray, None)
kp_r, des_r = sift.detectAndCompute(right_gray, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des_l, des_r, k=2)

# Using 0.8 instead of 0.5 to force more matches for the 40-to-50 gap
good_matches = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good_matches.append(list(kp_l[m.queryIdx].pt + kp_r[m.trainIdx].pt))
matches_arr = np.array(good_matches)

print(f"SIFT found {len(matches_arr)} candidate matches.")

# 3. Global RANSAC (Required for the baseline boundary and APAP filtering)
def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        rows.append([0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]])
        rows.append([p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]])
    _, _, V = np.linalg.svd(np.array(rows))
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]

def get_error(points, H):
    num_points = len(points)
    p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    p2 = points[:, 2:4]
    est_p2_unnorm = np.dot(H, p1.T).T
    est_p2 = est_p2_unnorm[:, 0:2] / est_p2_unnorm[:, 2:3]
    return np.linalg.norm(p2 - est_p2, axis=1) ** 2

best_inliers = []
best_global_H = np.eye(3)
max_inliers = 0

for _ in range(2000):
    idx = random.sample(range(len(matches_arr)), 4)
    H = homography(matches_arr[idx])
    if np.linalg.matrix_rank(H) < 3: continue
    
    errors = get_error(matches_arr, H)
    inliers = matches_arr[np.where(errors < 5.0)[0]]
    if len(inliers) > max_inliers:
        max_inliers = len(inliers)
        best_inliers = inliers
        best_global_H = H

print(f"RANSAC found {len(best_inliers)} Inliers for APAP.")

# 4. APAP (Moving DLT) Mesh Warping
def apap_stitch(left_img, right_img, inliers, global_H):
    if len(inliers) < 10:
        print("Not enough inliers for APAP mesh. Aborting.")
        return None

    h_l, w_l = left_img.shape[:2]
    h_r, w_r = right_img.shape[:2]
    
    # Calculate bounding box using Global H
    corners = np.float32([[0,0], [0,h_l], [w_l,h_l], [w_l,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, global_H)
    all_corners = np.concatenate((warped_corners, np.float32([[0,0], [0,h_r], [w_r,h_r], [w_r,0]]).reshape(-1,1,2)), axis=0)
    
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    out_w, out_h = x_max - x_min, y_max - y_min
    
    # --- APAP Mesh Setup ---
    GRID_SIZE = 25 # Calculate local H every 25 pixels
    grid_x = np.arange(0, out_w, GRID_SIZE)
    grid_y = np.arange(0, out_h, GRID_SIZE)
    
    map_x = np.zeros((len(grid_y), len(grid_x)), dtype=np.float32)
    map_y = np.zeros((len(grid_y), len(grid_x)), dtype=np.float32)
    
    src_pts = inliers[:, 0:2]
    dst_pts = inliers[:, 2:4]
    
    # Pre-build A matrix base
    N = len(inliers)
    A = np.zeros((2*N, 9))
    for i in range(N):
        u, v = src_pts[i]
        u_, v_ = dst_pts[i]
        A[2*i] = [-u, -v, -1, 0, 0, 0, u*u_, v*u_, u_]
        A[2*i+1] = [0, 0, 0, -u, -v, -1, u*v_, v*v_, v_]

    sigma, gamma = 8.5, 0.01
    
    print("Calculating APAP Mesh...")
    for i, gy in enumerate(tqdm(grid_y)):
        for j, gx in enumerate(grid_x):
            target_pt = np.array([gx + x_min, gy + y_min])
            
            # Distance-based Gaussian Weights
            dists = np.sum((dst_pts - target_pt)**2, axis=1)
            weights = np.maximum(np.exp(-dists / (sigma**2)), gamma)
            
            # Apply weights and solve SVD
            WA = A * np.repeat(weights, 2)[:, np.newaxis]
            _, _, V = np.linalg.svd(WA)
            local_H = V[-1].reshape(3, 3)
            
            # Inverse map to find source pixel for cv2.remap
            local_H_inv = np.linalg.inv(local_H)
            src_pt = np.dot(local_H_inv, [target_pt[0], target_pt[1], 1.0])
            src_pt /= src_pt[2]
            
            map_x[i, j] = src_pt[0]
            map_y[i, j] = src_pt[1]
            
    # Interpolate maps to full resolution
    map_x_full = cv2.resize(map_x, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    map_y_full = cv2.resize(map_y, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    
    # Mesh Warp Left Image, standard Warp Right Image
    warped_left = cv2.remap(left_rgb, map_x_full, map_y_full, cv2.INTER_LINEAR)
    warped_right = cv2.warpPerspective(right_rgb, T, (out_w, out_h))
    
    # Basic overlapping blend
    mask_l = np.any(warped_left > 0, axis=-1)
    mask_r = np.any(warped_right > 0, axis=-1)
    
    result = warped_left.copy()
    result[mask_r & ~mask_l] = warped_right[mask_r & ~mask_l]
    overlap = mask_l & mask_r
    result[overlap] = (warped_left[overlap].astype(int) + warped_right[overlap].astype(int)) // 2
    
    return result

# 5. Execute
apap_panorama = apap_stitch(left_rgb, right_rgb, best_inliers, best_global_H)

if apap_panorama is not None:
    plt.imshow(apap_panorama)
    plt.axis('off')
    plt.title("SIFT + APAP (Mesh Warping)")
    plt.show()
