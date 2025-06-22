# theme_check.py

from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from PIL import Image
import cv2

# Load CLIP model and processor once globally
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def cv2_to_pil(cv_img):
    """Convert OpenCV image to PIL."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def get_clip_embedding(image_pil):
    """Get L2-normalized CLIP embedding for an image."""
    inputs = clip_processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding.squeeze().cpu().numpy()

def cosine_delta(a, b):
    """1 - cosine similarity."""
    a = a / np.linalg.norm(a) if np.linalg.norm(a) else a
    b = b / np.linalg.norm(b) if np.linalg.norm(b) else b
    return 1 - np.dot(a, b)

def check_theme_difference(golden_img, actual_img, report_dict, threshold=0.1):
    """
    Detect major screen-level visual difference (like light/dark theme) using CLIP embeddings.

    Args:
        golden_img (np.ndarray): OpenCV image of golden screenshot.
        actual_img (np.ndarray): OpenCV image of actual screenshot.
        report_dict (dict): Reference to the main report to update.
        threshold (float): Cosine delta threshold to flag theme/background changes.

    Returns:
        None: Updates report_dict in-place.
    """
    golden_pil = cv2_to_pil(golden_img)
    actual_pil = cv2_to_pil(actual_img)

    emb_golden = get_clip_embedding(golden_pil)
    emb_actual = get_clip_embedding(actual_pil)

    delta = cosine_delta(emb_golden, emb_actual)
    explanation = (
        "Theme difference or light/dark mode or background style change"
        if delta > threshold else "No significant theme difference"
    )

    report_dict["overall_theme_issue"] = explanation
    print(f"🖼️ Screen delta: {round(delta, 4)} → {explanation}")
