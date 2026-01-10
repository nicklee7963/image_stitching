# Introduction
## What is feature detection
find specific parts of the image that can be used to track over time, ideally invariant to rotation, scale, translation

e.g. SURF, SIFT, ORB, BRIEF, FAST -> sparse feature detection

DINOV2 by nature is a **dense feature detector** can also be used as sparse feature detection. 

- **Sparse feature detection**: Looks for "Keypoints" or "Interest Points."
- **Dense feature detection**: describe the image using a uniform grid to capture the overall texture and content

## What is feature matching
Match the feature between two images, typically of different views

common methods uses **KNN**


## What is DINIv2
Paper: Learning Robust Visual Features without Supervision

## Why use DINOv2
- **Self-supervised learning**: doesn't need images with labels, can learn features in unseen images
- **No fine-tuning required**: can be used as backbone in many CV architectures
- **Diverse applications**: depth estimation, semantic segmantation, instance retrival(find similar image), dense matching, sparse matching.

## How does DINOv2 work? 
- **Vision Transformer(ViT)**

