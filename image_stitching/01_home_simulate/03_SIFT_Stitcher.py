import cv2
import matplotlib.pyplot as plt

# Set global plot size
plt.rcParams['figure.figsize'] = [15, 10]

left_img = cv2.imread('image/tv_desk/40.jpg')
right_img = cv2.imread('image/tv_desk/50.jpg')

if left_img is None or right_img is None:
    print("Error: Could not load one or both images.")
else:
    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    # 1. THE CRITICAL FIX: Use SCANS instead of PANORAMA
    stitcher = cv2.Stitcher_create(cv2.STITCHER_SCANS)

    print("Stitching images... (Using SCANS mode to prevent memory crash)")
    
    # 2. Perform the stitch
    status, panorama = stitcher.stitch([left_rgb, right_rgb])

    # 3. Check status
    if status == cv2.Stitcher_OK:
        print("Stitching successful!")
        plt.imshow(panorama)
        plt.axis('off')
        plt.title("OpenCV Stitcher (SCANS Mode - No Memory Crash)")
        plt.show()
    else:
        error_messages = {
            cv2.STITCHER_ERR_NEED_MORE_IMGS: "Error: Not enough matching points.",
            cv2.STITCHER_ERR_HOMOGRAPHY_EST_FAIL: "Error: Could not calculate geometry.",
            cv2.STITCHER_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Error: Failed to estimate camera parameters."
        }
        print(f"Stitching failed! Status code: {status}")
        print(error_messages.get(status, "Unknown error occurred."))
