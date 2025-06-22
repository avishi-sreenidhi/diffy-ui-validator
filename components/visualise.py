import cv2
import numpy as np

GREEN = (0, 255, 0)     
RED = (147, 20, 255)    
YELLOW = (0, 255, 255)  
ORANGE = (0, 165, 255)  

BOX_THICKNESS = 3

def draw_box(img, bbox, color, label=None):
    """Draws a bounding box with optional label."""
    if not bbox or len(bbox) != 4:
        return
    x, y, w, h = map(int, bbox)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, BOX_THICKNESS)
    if label:
        text_thickness = 2
        font_scale = 0.6
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        text_origin = (x, y - 10 if y - 10 > text_height else y + h + 20)
        cv2.putText(img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)

def visualize_report_differences(golden_img, actual_img, report):
    """Returns two images: annotated golden and annotated actual."""
    golden_rgb = cv2.cvtColor(golden_img, cv2.COLOR_BGR2RGB)
    actual_rgb = cv2.cvtColor(actual_img, cv2.COLOR_BGR2RGB)

    golden_annotated = golden_rgb.copy()
    actual_annotated = actual_rgb.copy()

    for match in report.get("matches", []):
        has_mismatch = bool(match.get("mismatch_type"))
        color = RED if has_mismatch else GREEN
        label = "Mismatch" if has_mismatch else "Match"

        draw_box(golden_annotated, match.get("golden", {}).get("bbox"), color, label)
        draw_box(actual_annotated, match.get("actual", {}).get("bbox"), color, label)

    for item in report.get("unmatched_golden", []):
        bbox = item.get("bbox") or item.get("golden", {}).get("bbox")  
        draw_box(golden_annotated, bbox, YELLOW, label="Missing")

    for item in report.get("unmatched_actual", []):
        bbox = item.get("bbox") or item.get("actual", {}).get("bbox")  
        draw_box(actual_annotated, bbox, ORANGE, label="Extra")

    return golden_annotated, actual_annotated
