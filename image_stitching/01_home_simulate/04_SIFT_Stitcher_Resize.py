import cv2
import matplotlib.pyplot as plt

# Set global plot size
plt.rcParams['figure.figsize'] = [15, 10]

# 1. Load Images
def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot find image at {path}")
    # Convert immediately to RGB for Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

# 2. Resize Function (Crucial for preventing SHRT_MAX and OOM crashes)
def resize_image(img, max_size=800):
    """Resizes the image so its longest side is max_size."""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h))
    return img

# Load your images
left_rgb = read_image('image/tv_desk/40.jpg')
# You can change this to 50.jpg or 70.jpg
right_rgb = read_image('image/tv_desk/50.jpg') 

# Resize images BEFORE passing to the Stitcher
left_rgb = resize_image(left_rgb)
right_rgb = resize_image(right_rgb)

# 3. Initialize OpenCV Stitcher in SCANS mode
print("Initializing OpenCV Stitcher (SCANS mode)...")
# SCANS mode prevents the 3D spherical math from blowing up your memory
stitcher = cv2.Stitcher_create(cv2.STITCHER_SCANS)

# 4. Perform the stitch
print("Stitching images... (OpenCV is handling feature matching and blending)")
status, panorama = stitcher.stitch([left_rgb, right_rgb])

# 5. Check status and Display
if status == cv2.Stitcher_OK:
    print("Stitching successful!")
    plt.imshow(panorama)
    plt.axis('off')
    plt.title("OpenCV Stitcher (SCANS Mode with Resized Inputs)")
    plt.show()
else:
    error_messages = {
        cv2.STITCHER_ERR_NEED_MORE_IMGS: "Error: Not enough matching points. Images might not overlap enough.",
        cv2.STITCHER_ERR_HOMOGRAPHY_EST_FAIL: "Error: Could not calculate geometry. The angle is too extreme.",
        cv2.STITCHER_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Error: Failed to estimate camera parameters."
    }
    print(f"Stitching failed! Status code: {status}")
    print(error_messages.get(status, "Unknown error occurred."))
