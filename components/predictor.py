

from roboflow import Roboflow
import os
from dotenv import load_dotenv


load_dotenv()


api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("Missing ROBOFLOW_API_KEY in .env file")

rf = Roboflow(api_key=api_key)
project = rf.workspace("uied").project("ui-element-detect")
model = project.version(6).model

def predict_ui_elements(image_path_or_cv2):
    """
    Accepts a path to an image OR a cv2 image array and returns parsed bounding boxes.
    """
    try:
        prediction = model.predict(image_path_or_cv2).json()
        return parse_predictions(prediction)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

def parse_predictions(predictions):
    """
    Converts raw predictions into simplified class + bounding box format.
    Format: {'class': str, 'bbox': [x, y, width, height]}
    """
    parsed = []
    for item in predictions.get('predictions', []):
        parsed.append({
            "class": item.get("class"),
            "bbox": [
                int(item["x"] - item["width"] / 2),
                int(item["y"] - item["height"] / 2),
                int(item["width"]),
                int(item["height"])
            ]
        })
    return parsed
