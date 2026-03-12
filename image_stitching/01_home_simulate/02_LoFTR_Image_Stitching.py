import cv2
import torch
import kornia as K
from kornia.feature import LoFTR
import matplotlib.pyplot as plt
import numpy as np

# Set plot size
plt.rcParams['figure.figsize'] = [15, 10]

# 1. Load Images
def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot find image at {path}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb

 # --- Add this right after your read_image function ---
def resize_for_loftr(img, max_size=640):
    """Resizes image so the longest edge is max_size, and dimensions are multiples of 8."""
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    
    # Calculate new dimensions and force them to be multiples of 8
    new_w = int(w * scale) // 8 * 8
    new_h = int(h * scale) // 8 * 8
    
    return cv2.resize(img, (new_w, new_h))

# --- Update your Step 1 loading logic to this ---
left_gray, _, left_rgb = read_image('image/tv_desk/40.jpg')
right_gray, _, right_rgb = read_image('image/tv_desk/70.jpg')

# Resize all images BEFORE passing to LoFTR
left_gray = resize_for_loftr(left_gray)
right_gray = resize_for_loftr(right_gray)
left_rgb = resize_for_loftr(left_rgb)
right_rgb = resize_for_loftr(right_rgb)   

# 2. Prepare Tensors for PyTorch / Kornia
# LoFTR needs grayscale images in the shape (Batch, Channels, Height, Width)
# normalized between 0.0 and 1.0 as floats.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running LoFTR on: {device}")

img_left_tensor = K.image_to_tensor(left_gray, keepdim=False).float() / 255.0
img_right_tensor = K.image_to_tensor(right_gray, keepdim=False).float() / 255.0

img_left_tensor = img_left_tensor.to(device)
img_right_tensor = img_right_tensor.to(device)

# 3. Initialize LoFTR
# We use the 'indoor' weights because a desk/TV scene is an indoor environment
matcher = LoFTR(pretrained='indoor').to(device)

# 4. Perform Matching
print("Extracting and matching features with LoFTR...")
input_dict = {
    "image0": img_left_tensor, 
    "image1": img_right_tensor
}

with torch.no_grad():
    correspondences = matcher(input_dict)

# Extract matched coordinates and convert back to NumPy for OpenCV
# mkpts0 = matched keypoints in image 0 (left)
# mkpts1 = matched keypoints in image 1 (right)
mkpts_left = correspondences['keypoints0'].cpu().numpy()
mkpts_right = correspondences['keypoints1'].cpu().numpy()

print(f"LoFTR found {len(mkpts_left)} high-quality matches!")

# 5. Calculate Homography (Using OpenCV's modern MAGSAC++ instead of basic RANSAC)
# MAGSAC is much more robust against parallax errors than standard RANSAC
H_matrix, mask = cv2.findHomography(mkpts_left, mkpts_right, cv2.USAC_MAGSAC, 3.0)
inliers = mask.ravel().tolist()
print(f"Homography Inliers: {sum(inliers)}/{len(mkpts_left)}")

# 6. Fast NumPy Stitching (No manual loops!)
def fast_stitch(left_img, right_img, H):
    print("Stitching images using fast NumPy matrix operations...")
    h_l, w_l = left_img.shape[:2]
    h_r, w_r = right_img.shape[:2]
    
    # Find bounding box
    corners_left = np.float32([[0, 0], [0, h_l], [w_l, h_l], [w_l, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_left, H)
    all_corners = np.concatenate((warped_corners, np.float32([[0, 0], [0, h_r], [w_r, h_r], [w_r, 0]]).reshape(-1, 1, 2)), axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]], dtype=np.float32)
    H = H.astype(np.float32)

    output_size = (x_max - x_min, y_max - y_min)
    
    # Warp both images
    warped_l = cv2.warpPerspective(left_img, np.dot(H_translation, H), output_size)
    warped_r = cv2.warpPerspective(right_img, H_translation, output_size)

    # Fast Blending
    mask_l = np.any(warped_l > 0, axis=-1)
    mask_r = np.any(warped_r > 0, axis=-1)
    
    result = warped_l.copy()
    result[mask_r & ~mask_l] = warped_r[mask_r & ~mask_l]
    
    # Simple averaging for overlap (you will still see minor ghosting if parallax is severe)
    overlap = mask_l & mask_r
    result[overlap] = (warped_l[overlap].astype(int) + warped_r[overlap].astype(int)) // 2
    
    return result.astype(np.uint8)

# 7. Execute and Display
final_result = fast_stitch(left_rgb, right_rgb, H_matrix)

plt.imshow(final_result)
plt.axis('off')
plt.title("LoFTR Panorama Stitching")
plt.show()
