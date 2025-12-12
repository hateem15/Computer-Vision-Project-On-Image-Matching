"""
Simple example: Match two images using LightGlue

This is a minimal example showing how to use LightGlue for image matching.
Now, instead of editing the script, you can type the image names in the console
when you run it (e.g. `1`, `2`, `3`, etc. for files in the `mohid` folder).
"""

import os
import sys

import torch
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ============================================
# CHECK TORCH INSTALLATION
# ============================================
print("="*50)
print("Checking PyTorch installation...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
print("="*50)
print()

# ============================================
# CONFIGURATION - Image selection
# ============================================
# All images are expected to be inside this folder
IMAGES_FOLDER = "mohid"

# Allowed image file extensions (you can add more if needed)
ALLOWED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def resolve_image_path(name: str) -> str:
    """
    Turn a user-provided name into a valid image path.

    - If the user includes an extension (e.g. `1.jpg`), use it directly.
    - If the user omits the extension (e.g. `1`), search for any allowed
      extension in the `IMAGES_FOLDER`.
    """
    name = name.strip()
    if not name:
        print("Image name cannot be empty.")
        sys.exit(1)

    # If user already typed an extension, just check that exact file
    root, ext = os.path.splitext(name)
    if ext:
        candidate = os.path.join(IMAGES_FOLDER, name)
        if os.path.exists(candidate):
            return candidate
        print(f"File '{candidate}' not found.")
        sys.exit(1)

    # No extension given: try all allowed extensions
    for e in ALLOWED_EXTS:
        candidate = os.path.join(IMAGES_FOLDER, root + e)
        if os.path.exists(candidate):
            return candidate

    print(
        f"No file found for '{name}' in '{IMAGES_FOLDER}' with extensions "
        f"{', '.join(ALLOWED_EXTS)}."
    )
    sys.exit(1)


print("=" * 50)
print(f"Images folder: '{IMAGES_FOLDER}'")
print("Enter image names that exist in this folder.")
print("Examples: 1, 2, 3, 4 or with extension like 1.jpg, 3.png")
print("=" * 50)

img0_name = input("Image 0 name (e.g. 1): ")
img1_name = input("Image 1 name (e.g. 2): ")

IMAGE0_PATH = resolve_image_path(img0_name)
IMAGE1_PATH = resolve_image_path(img1_name)

# Feature extractor options: 'superpoint', 'disk', 'sift', 'aliked'
FEATURE_TYPE = 'superpoint'
MAX_KEYPOINTS = 2048  # Reduce for faster processing (e.g., 1024)

# Visualization settings
MAX_MATCHES_TO_SHOW = 20  # Limit number of matches to display (set to None to show all)
LINE_OPACITY = 0.5  # Make lines more visible (0.0 to 1.0)

# ============================================
# CODE - No need to modify below
# ============================================

# Set device (automatically uses GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load feature extractor
print(f"Loading {FEATURE_TYPE} extractor...")
extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(device)

# Load LightGlue matcher
print(f"Loading LightGlue matcher...")
matcher = LightGlue(features=FEATURE_TYPE).eval().to(device)

# Load images
print(f"Loading images...")
image0 = load_image(IMAGE0_PATH).to(device)
image1 = load_image(IMAGE1_PATH).to(device)

# Extract local features from both images
print("Extracting features...")
feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)

print(f"  Image 0: {len(feats0['keypoints'][0])} keypoints")
print(f"  Image 1: {len(feats1['keypoints'][0])} keypoints")

# Match the features
print("Matching features...")
matches01 = matcher({'image0': feats0, 'image1': feats1})

# Remove batch dimension
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

# Extract matched keypoints
kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
m_kpts0 = kpts0[matches[..., 0]]  # matched keypoints in image 0
m_kpts1 = kpts1[matches[..., 1]]  # matched keypoints in image 1

num_matches = len(matches)
print(f"\n✓ Found {num_matches} matches!")

if 'stop' in matches01:
    print(f"  LightGlue stopped after {matches01['stop']} layers")

# Visualize the matches with lines connecting matched points
print("\nDisplaying visualization with match lines...")
axes = viz2d.plot_images([image0.cpu(), image1.cpu()], 
                         titles=['Image 0', 'Image 1'])

# Limit the number of matches to display for better visibility
if MAX_MATCHES_TO_SHOW is not None and num_matches > MAX_MATCHES_TO_SHOW:
    # Sample matches evenly
    indices = np.linspace(0, num_matches - 1, MAX_MATCHES_TO_SHOW, dtype=int)
    display_kpts0 = m_kpts0[indices]
    display_kpts1 = m_kpts1[indices]
    print(f"  Showing {MAX_MATCHES_TO_SHOW} of {num_matches} matches for clarity")
else:
    display_kpts0 = m_kpts0
    display_kpts1 = m_kpts1

# Generate different colors for each match line
num_display_matches = len(display_kpts0)
# Use a colormap to generate distinct colors
colormap = cm.get_cmap('hsv')  # 'hsv' gives good color distribution
colors = [colormap(i / num_display_matches) for i in range(num_display_matches)]

# Draw lines between matched points with different colors for each line
viz2d.plot_matches(display_kpts0.cpu(), display_kpts1.cpu(), 
                   color=colors, lw=1.2, ps=5, a=LINE_OPACITY)

# Add text overlay with match count
viz2d.add_text(0, f'{num_matches} matches (showing {len(display_kpts0)})', fs=20)

if 'stop' in matches01:
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=15)

plt.tight_layout()
plt.show()

# Access the matched points as numpy arrays if needed
matched_points_0 = m_kpts0.cpu().numpy()  # shape: (num_matches, 2)
matched_points_1 = m_kpts1.cpu().numpy()  # shape: (num_matches, 2)

print(f"\nMatched points shape: {matched_points_0.shape}")
print(f"First 5 matches in image 0: {matched_points_0[:5]}")
print(f"First 5 matches in image 1: {matched_points_1[:5]}")

