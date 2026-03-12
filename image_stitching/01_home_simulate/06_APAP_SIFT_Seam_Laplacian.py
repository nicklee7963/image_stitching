import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['figure.figsize'] = [15, 10]

# ==========================================
# 1. I/O & SIFT MATCHING
# ==========================================
def read_and_resize(path, max_dim=800):
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

left_gray, left_rgb = read_and_resize('image/tv_desk/40.jpg')
right_gray, right_rgb = read_and_resize('image/tv_desk/70.jpg')

try:
    sift = cv2.SIFT_create(nfeatures=5000)
except AttributeError:
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=5000)

kp_l, des_l = sift.detectAndCompute(left_gray, None)
kp_r, des_r = sift.detectAndCompute(right_gray, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des_l, des_r, k=2)

# Aggressive ratio test to ensure enough points for APAP
good_matches = [list(kp_l[m.queryIdx].pt + kp_r[m.trainIdx].pt) 
                for m, n in matches if m.distance < 0.8 * n.distance]
matches_arr = np.array(good_matches)

print(f"SIFT found {len(matches_arr)} candidate matches.")

# ==========================================
# 2. GLOBAL RANSAC (For Canvas Sizing)
# ==========================================
def get_global_homography(matches_arr):
    src_pts = matches_arr[:, 0:2].reshape(-1, 1, 2)
    dst_pts = matches_arr[:, 2:4].reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = matches_arr[mask.ravel() == 1]
    return inliers, H

inliers, global_H = get_global_homography(matches_arr)
print(f"RANSAC found {len(inliers)} Inliers.")

# ==========================================
# 3. APAP MESH WARPING
# ==========================================

def build_apap_matrices(inliers, out_w, out_h, x_min, y_min):
    GRID_SIZE = 25 
    grid_x = np.arange(0, out_w, GRID_SIZE)
    grid_y = np.arange(0, out_h, GRID_SIZE)
    
    map_x = np.zeros((len(grid_y), len(grid_x)), dtype=np.float32)
    map_y = np.zeros((len(grid_y), len(grid_x)), dtype=np.float32)
    
    src_pts, dst_pts = inliers[:, 0:2], inliers[:, 2:4]
    
    N = len(inliers)
    A = np.zeros((2*N, 9))
    for i in range(N):
        u, v = src_pts[i]
        u_, v_ = dst_pts[i]
        A[2*i] = [-u, -v, -1, 0, 0, 0, u*u_, v*u_, u_]
        A[2*i+1] = [0, 0, 0, -u, -v, -1, u*v_, v*v_, v_]

    print("Calculating APAP Mesh...")
    for i, gy in enumerate(tqdm(grid_y)):
        for j, gx in enumerate(grid_x):
            target_pt = np.array([gx + x_min, gy + y_min])
            dists = np.sum((dst_pts - target_pt)**2, axis=1)
            weights = np.maximum(np.exp(-dists / (8.5**2)), 0.01)
            
            WA = A * np.repeat(weights, 2)[:, np.newaxis]
            _, _, V = np.linalg.svd(WA)
            local_H = V[-1].reshape(3, 3)
            
            local_H_inv = np.linalg.inv(local_H)
            src_pt = np.dot(local_H_inv, [target_pt[0], target_pt[1], 1.0])
            map_x[i, j] = src_pt[0] / src_pt[2]
            map_y[i, j] = src_pt[1] / src_pt[2]
            
    return cv2.resize(map_x, (out_w, out_h)), cv2.resize(map_y, (out_w, out_h))

h_l, w_l = left_rgb.shape[:2]
h_r, w_r = right_rgb.shape[:2]

corners = np.float32([[0,0], [0,h_l], [w_l,h_l], [w_l,0]]).reshape(-1,1,2)
warped_corners = cv2.perspectiveTransform(corners, global_H)
all_corners = np.concatenate((warped_corners, np.float32([[0,0], [0,h_r], [w_r,h_r], [w_r,0]]).reshape(-1,1,2)), axis=0)

x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
out_w, out_h = x_max - x_min, y_max - y_min

T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

map_x_full, map_y_full = build_apap_matrices(inliers, out_w, out_h, x_min, y_min)

warped_left = cv2.remap(left_rgb, map_x_full, map_y_full, cv2.INTER_LINEAR)
warped_right = cv2.warpPerspective(right_rgb, T, (out_w, out_h))

# Create masks (255 where image exists, 0 where black)
mask_l = (cv2.cvtColor(warped_left, cv2.COLOR_RGB2GRAY) > 0).astype(np.uint8) * 255
mask_r = (cv2.cvtColor(warped_right, cv2.COLOR_RGB2GRAY) > 0).astype(np.uint8) * 255

# Ensure dimensions are divisible by 16 for the Pyramid Blending
pad_h = (16 - out_h % 16) % 16
pad_w = (16 - out_w % 16) % 16
warped_left = cv2.copyMakeBorder(warped_left, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
warped_right = cv2.copyMakeBorder(warped_right, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
mask_l = cv2.copyMakeBorder(mask_l, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
mask_r = cv2.copyMakeBorder(mask_r, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

# ==========================================
# 4. VORONOI SEAM FINDING (Graph Cut Approximation)
# ==========================================

print("Calculating Optimal Seam...")
# Distance transform finds how far every pixel is from the edge of its image
dist_l = cv2.distanceTransform(mask_l, cv2.DIST_L2, 5)
dist_r = cv2.distanceTransform(mask_r, cv2.DIST_L2, 5)

# The seam is drawn exactly where the distance from Image A's edge equals Image B's edge
seam_mask = (dist_l > dist_r).astype(np.float32)

# Smooth the seam mask slightly to prevent harsh stair-stepping
seam_mask = cv2.GaussianBlur(seam_mask, (5, 5), 0)
seam_mask = np.repeat(seam_mask[:, :, np.newaxis], 3, axis=2)

# ==========================================
# 5. LAPLACIAN MULTI-BAND BLENDING
# ==========================================

print("Applying Laplacian Multi-band Blending...")
def laplacian_blend(img1, img2, mask, levels=4):
    # 1. Build Gaussian Pyramids
    G1, G2, GM = img1.astype(np.float32), img2.astype(np.float32), mask.copy()
    gp1, gp2, gpM = [G1], [G2], [GM]
    
    for _ in range(levels):
        G1, G2, GM = cv2.pyrDown(G1), cv2.pyrDown(G2), cv2.pyrDown(GM)
        gp1.append(G1); gp2.append(G2); gpM.append(GM)
        
    # 2. Build Laplacian Pyramids
    lp1, lp2 = [gp1[levels]], [gp2[levels]]
    for i in range(levels, 0, -1):
        # Resize to handle odd-dimension rounding issues
        h, w = gp1[i-1].shape[:2]
        L1 = cv2.subtract(gp1[i-1], cv2.resize(cv2.pyrUp(gp1[i]), (w, h)))
        L2 = cv2.subtract(gp2[i-1], cv2.resize(cv2.pyrUp(gp2[i]), (w, h)))
        lp1.append(L1); lp2.append(L2)
        
    # 3. Blend Pyramids based on the Seam Mask
    LS = []
    for l1, l2, gm in zip(lp1, lp2, reversed(gpM)):
        h, w = l1.shape[:2]
        gm_resized = cv2.resize(gm, (w, h))
        ls = l1 * gm_resized + l2 * (1.0 - gm_resized)
        LS.append(ls)
        
    # 4. Reconstruct Final Image
    ls_ = LS[0]
    for i in range(1, levels + 1):
        h, w = LS[i].shape[:2]
        ls_ = cv2.add(cv2.resize(cv2.pyrUp(ls_), (w, h)), LS[i])
        
    return np.clip(ls_, 0, 255).astype(np.uint8)

final_panorama = laplacian_blend(warped_left, warped_right, seam_mask, levels=5)

# Display Output
plt.imshow(final_panorama)
plt.axis('off')
plt.title("Professional Stitch (APAP + Voronoi Seam + Laplacian Blending)")
plt.show()
