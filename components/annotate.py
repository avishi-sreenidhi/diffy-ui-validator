import cv2

def annotate_image(image, components, is_golden=False):
    annotated = image.copy()
    for comp in components:
        bbox = comp.get("bbox")
        if not bbox:
            print("Skipping component without bbox:", comp)
            continue

        if len(bbox) == 4:
            x, y, w, h = bbox
            if w > 1 and h > 1 and (w < 100 and h < 100):
                pass
            else:
                x1, y1, x2, y2 = bbox
                x, y = x1, y1
                w, h = x2 - x1, y2 - y1
        else:
            print(f"Invalid bbox length {len(bbox)} for comp:", comp)
            continue

        x = max(0, int(x))
        y = max(0, int(y))
        w = max(1, int(w))
        h = max(1, int(h))
        if x + w > image.shape[1]:
            w = image.shape[1] - x
        if y + h > image.shape[0]:
            h = image.shape[0] - y

        label = comp.get("class", "Unknown")
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text = comp.get("text_content", "")
        display_text = label
        if text:
            display_text += f": {text[:30]}"

        cv2.putText(annotated, display_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    return annotated 


def save_annotated_images(annotated_golden, annotated_error,
                          golden_path="golden_annotated.png",
                          error_path="error_annotated.png"):
    """
    Save annotated images to disk.

    Args:
        annotated_golden (np.ndarray): Annotated golden image.
        annotated_error (np.ndarray): Annotated error image.
        golden_path (str): File path to save golden annotated image.
        error_path (str): File path to save error annotated image.
    """
    cv2.imwrite(golden_path, annotated_golden)
    cv2.imwrite(error_path, annotated_error)
