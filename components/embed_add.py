import numpy as np
import torch
import cv2
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from components.ocr_add import extract_roi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Load Models 
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_model = SentenceTransformer("all-MiniLM-L6-v2")

clip_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

def extract_ui_features(image, bbox):
    roi = extract_roi(image, bbox)
    if roi.size == 0:
        return None

    mean_color = np.mean(roi, axis=(0, 1))
    std_color = np.std(roi, axis=(0, 1))
    h, w = roi.shape[:2]
    aspect_ratio = w / h if h != 0 else 0
    area = w * h

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)

    handcrafted = np.array([
        *mean_color, *std_color, aspect_ratio, area, edge_density
    ], dtype=np.float32)

    return handcrafted

def get_visual_embedding(image, bbox, clip_weight=0.8, ui_weight=0.3):
    roi = extract_roi(image, bbox)
    if roi.size == 0:
        return None

    try:
        pil_img = transforms.ToPILImage()(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            clip_feat = clip_model.get_image_features(**inputs).squeeze().cpu().numpy()

        ui_feat = extract_ui_features(image, bbox)
        if ui_feat is None:
            ui_feat = np.zeros(8, dtype=np.float32)

        clip_feat = clip_feat / np.linalg.norm(clip_feat) if np.linalg.norm(clip_feat) > 1e-8 else clip_feat
        ui_feat = ui_feat / np.linalg.norm(ui_feat) if np.linalg.norm(ui_feat) > 1e-8 else ui_feat

        clip_feat *= clip_weight
        ui_feat *= ui_weight

        fused = np.concatenate([clip_feat, ui_feat])
        fused = fused / np.linalg.norm(fused) if np.linalg.norm(fused) > 1e-8 else fused

        return fused

    except Exception as e:
        print(f"⚠️ CLIP embedding error: {e}")
        return None

def get_text_embedding(text):
    return text_model.encode(text) if text else np.zeros(384)

def get_class_embedding(cls):
    return text_model.encode(cls)

def get_spatial_embedding(bbox, image_shape):
    h, w = image_shape[:2]
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2, y + bh / 2
    return np.array([cx / w, cy / h, bw / w, bh / h], dtype=np.float32)

def fuse_embeddings(visual, text, spatial, cls, weights=None):
    if weights is None:
        weights = {"visual": 10.5, "text": 1.0, "spatial": 1.0, "class": 1.0}

    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-8 else v

    vectors = []
    if visual is not None:
        vectors.append(normalize(visual) * weights["visual"])
    else:
        print("⚠️ Missing visual embedding")
    if text is not None:
        vectors.append(normalize(text) * weights["text"])
    if spatial is not None:
        vectors.append(normalize(spatial) * weights["spatial"])
    if cls is not None:
        vectors.append(normalize(cls) * weights["class"])

    return np.concatenate(vectors) if vectors else np.zeros(8)

def enrich_components(image, components, image_shape):
    enriched = []
    for comp in components:
        bbox = comp["bbox"]

        visual = get_visual_embedding(image, bbox)
        text_vec = get_text_embedding(comp.get("text_content", ""))
        spatial = get_spatial_embedding(bbox, image_shape)
        cls_vec = get_class_embedding(comp.get("class", "unknown"))

        comp["embedding_visual"] = visual.tolist() if visual is not None else None
        comp["embedding_text"] = text_vec.tolist()
        comp["embedding_spatial"] = spatial.tolist()
        comp["embedding_class"] = cls_vec.tolist()

        if visual is not None:
            comp["embedding_fused"] = fuse_embeddings(visual, text_vec, spatial, cls_vec).tolist()
        else:
            comp["embedding_fused"] = None

        enriched.append(comp)
    return enriched
