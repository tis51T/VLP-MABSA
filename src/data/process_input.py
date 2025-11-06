import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torchvision.ops import roi_align
import torch.nn as nn
import json

# ──────────────────────────────────────────────────────
# 1. Image and Text Loading Utilities
# ──────────────────────────────────────────────────────
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def load_text(json_path, img_id):
    """Load text (e.g., tweet) associated with image ID."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(img_id, "")

# ──────────────────────────────────────────────────────
# 2. Faster R-CNN Feature Extraction Utilities
# ──────────────────────────────────────────────────────
def prepare_faster_rcnn_model(device="cuda"):
    # Use ResNet-101 backbone with FPN from torchvision
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()
    return model

def extract_region_features(image, model, transform, device="cuda", num_regions=36):
    # Transform image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Run detection
    with torch.no_grad():
        outputs = model(img_tensor)[0]

    # Get boxes and scores
    boxes = outputs["boxes"]
    scores = outputs["scores"]

    if len(boxes) == 0:
        return np.zeros((num_regions, 2048)), np.zeros((num_regions, 4))

    # Select top-N detections
    top_indices = scores.topk(k=min(num_regions, len(scores))).indices
    selected_boxes = boxes[top_indices]

    # Prepare ROI Align (need batch index for each box)
    box_indices = torch.zeros((len(selected_boxes),), dtype=torch.int64).to(device)
    rois = torch.cat([box_indices[:, None], selected_boxes], dim=1)

    # Extract FPN feature maps
    with torch.no_grad():
        feature_maps = model.backbone(img_tensor)  # dict of feature maps from FPN

    # Use the highest resolution feature map (often '0')
    fmap = list(feature_maps.values())[0]  # using first key's feature map from FPN

    # ROI Align: output size = (7, 7)
    pooled = roi_align(fmap, rois, output_size=(7, 7), spatial_scale=1.0)

    # Global average pooling → (num_regions, 2048)
    features = pooled.mean(dim=[2, 3])

    return features.cpu().numpy(), selected_boxes.cpu().numpy()



# ──────────────────────────────────────────────────────
# 3. Optional: Project 2048-d Features to 768-d
# ──────────────────────────────────────────────────────
def project_features(region_features, output_dim=768, device=None):
    # dynamically handle input dim and device instead of assuming 2048/CUDA
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # region_features may be numpy array; convert to tensor on the chosen device
    region_tensor = torch.tensor(region_features, dtype=torch.float32, device=device)
    in_dim = region_tensor.shape[1] if region_tensor.ndim == 2 else int(region_tensor.numel())
    projection_layer = nn.Linear(in_dim, output_dim).to(device)
    with torch.no_grad():
        projected = projection_layer(region_tensor).cpu().numpy()
    return projected

# ──────────────────────────────────────────────────────
# 4. Batch Processing and Saving
# ──────────────────────────────────────────────────────
def process_and_save_features(image_dir, text_json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Define image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = prepare_faster_rcnn_model(device)

    # Process all images in directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_id = os.path.splitext(filename)[0]
            image_path = os.path.join(image_dir, filename)

            # Load image + text
            image = load_image(image_path)
            text = load_text(text_json_path, image_id)

            # Extract region features
            region_features, boxes = extract_region_features(image, model, transform, device)

            # Optionally project to 768-d (for BART or Transformer compatibility)
            region_features_768 = project_features(region_features, output_dim=768)

            # Save to .npy (just features)
            np.save(os.path.join(output_dir, f"{image_id[:-4]}.npy"), region_features_768)

            # Save to .npz (features + text + boxes)
            np.savez(os.path.join(output_dir, f"{image_id[:-4]}.npz"),
                     img_features=region_features_768,
                     boxes=boxes,
                     text=text)

            print(f"Saved features for {image_id}")

    print("✅ All images processed and saved!")
