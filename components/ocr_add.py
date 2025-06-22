import cv2
import pytesseract

def extract_roi(image, bbox):
    """
    Extracts the region of interest (ROI) from the image based on the bounding box.

    Args:
        image (np.array): Input image.
        bbox (tuple): Bounding box (x, y, w, h).

    Returns:
        np.array: Cropped ROI image.
    """
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

def preprocess_for_ocr(roi):
    """
    Preprocesses the ROI to improve OCR accuracy.

    Steps:
    - Convert to grayscale
    - Apply OTSU thresholding
    - Resize to enlarge text

    Args:
        roi (np.array): Region of interest.

    Returns:
        np.array: Preprocessed image for OCR.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    scale_factor = 2
    resized = cv2.resize(thresh, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    return resized

def perform_ocr_on_component(image, component):
    """
    Runs OCR on a single component bounding box.

    Skips OCR if component class suggests it's an image/icon.

    Args:
        image (np.array): Full input image.
        component (dict): Dict containing 'bbox' and 'class'.

    Returns:
        dict: Updated component with 'text' and 'text_content' fields.
    """
    label = component.get("class", "").lower()

    if "image" in label or "icon" in label:
        component["text"] = False
        component["text_content"] = None
        return component

    roi = extract_roi(image, component["bbox"])
    preprocessed_roi = preprocess_for_ocr(roi)

    text = pytesseract.image_to_string(preprocessed_roi, config='--psm 6').strip()
    component["text"] = bool(text)
    component["text_content"] = text if text else None

    return component

def add_ocr_data(image, components):
    """
    Applies OCR to all components.

    Args:
        image (np.array): Input image.
        components (list): List of component dicts.

    Returns:
        list: Updated components with OCR text info.
    """
    return [perform_ocr_on_component(image, comp) for comp in components]
