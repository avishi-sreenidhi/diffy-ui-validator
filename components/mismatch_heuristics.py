import numpy as np

def get_avg_color_hex(image, bbox):
    """Returns the average color in hex within the bounding box of the image."""
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    avg_color = roi.mean(axis=(0, 1))  
    r, g, b = int(avg_color[2]), int(avg_color[1]), int(avg_color[0])
    return f"#{r:02X}{g:02X}{b:02X}"


def enrich_with_heuristics(matches, golden_img, actual_img):
    """Adds heuristic explanations for mismatched components."""
    for match in matches:
        issues = []
        g, a = match["golden"], match["actual"]
        deltas = match.get("delta_scores", {})
        mismatch_types = match.get("mismatch_type", [])

        if "visual" in mismatch_types:
            visual_score = deltas.get("visual", 0)
            g_class = g.get("class", "").lower()
            a_class = a.get("class", "").lower()

            g_color = g.get("color_hex", "").lower() if g.get("color_hex") else None
            a_color = a.get("color_hex", "").lower() if a.get("color_hex") else None

            if not g_color:
                g_color = get_avg_color_hex(golden_img, g.get("bbox"))
            if not a_color:
                a_color = get_avg_color_hex(actual_img, a.get("bbox"))

            if visual_score > 0.8:
                issues.append("Visual embeddings differ drastically — possibly wrong image asset.")

            elif any(x in g_class for x in ["button", "textbutton", "container"]):
                if g_color and a_color:
                    if g_color != a_color:
                        issues.append(f"Button color mismatch — expected {g_color}, got {a_color}")
                    elif visual_score > 0.015:
                        issues.append("Component appearance changed — likely due to style/padding, also check text.")
                else:
                    missing = []
                    if not g_color: missing.append("design color")
                    if not a_color: missing.append("actual color")
                    issues.append(f"Component appearance changed — color comparison not possible (missing {', '.join(missing)}).")

        if "position" in mismatch_types:
            pos = deltas.get("position")
            shift_desc = []

            if isinstance(pos, dict):
                if pos.get("x_shift"): shift_desc.append(f"x: {pos['x_shift']}px")
                if pos.get("y_shift"): shift_desc.append(f"y: {pos['y_shift']}px")
                if pos.get("w_diff"): shift_desc.append(f"width: {pos['w_diff']}px")
                if pos.get("h_diff"): shift_desc.append(f"height: {pos['h_diff']}px")
            elif isinstance(pos, (int, float)):
                g_x, g_y, g_w, g_h = g.get("bbox", [0, 0, 0, 0])
                a_x, a_y, a_w, a_h = a.get("bbox", [0, 0, 0, 0])

                x_shift = a_x - g_x
                y_shift = a_y - g_y
                w_diff = a_w - g_w
                h_diff = a_h - g_h

                if x_shift: shift_desc.append(f"x: {x_shift:+d}px")
                if y_shift: shift_desc.append(f"y: {y_shift:+d}px")
                if w_diff: shift_desc.append(f"width: {w_diff:+d}px")
                if h_diff: shift_desc.append(f"height: {h_diff:+d}px")

            if shift_desc:
                issues.append("Layout shift detected — " + ", ".join(shift_desc))

        if "text" in mismatch_types:
            old_text = g.get("text_content", "")
            new_text = a.get("text_content", "")
            if old_text != new_text:
                issues.append(f"Text changed from '{old_text}' to '{new_text}'")

        if not issues and mismatch_types:
            issues.append("Mismatch detected, but no specific heuristic triggered.")

        match["heuristic_explanation"] = issues
