import numpy as np
from difflib import SequenceMatcher


def cosine_delta(a, b):
    a, b = np.array(a), np.array(b)
    a_norm = a / np.linalg.norm(a) if np.linalg.norm(a) else a
    b_norm = b / np.linalg.norm(b) if np.linalg.norm(b) else b
    return 1 - np.dot(a_norm, b_norm)


def text_delta(t1, t2):
    return 1 - SequenceMatcher(None, t1 or "", t2 or "").ratio()


def position_delta(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return abs(x1 - x2) + abs(y1 - y2) + abs(w1 - w2) + abs(h1 - h2)


def classify_mismatches(results, thresholds=None):
    """
    Annotate each match with types of mismatches based on visual, text, position, and class change.
    """
    if thresholds is None:
        thresholds = {
            "visual": 0.025,
            "text": 0.3,
            "spatial": 30
        }

    for match in results.get("matches", []):
        g = match["golden"]
        a = match["actual"]

        vis_delta = cosine_delta(g.get("embedding_visual", []), a.get("embedding_visual", []))
        txt_delta = text_delta(g.get("text_content", ""), a.get("text_content", ""))
        pos_delta = position_delta(g.get("bbox", [0, 0, 0, 0]), a.get("bbox", [0, 0, 0, 0]))
        class_changed = g.get("class") != a.get("class")

        mismatch_type = []
        if vis_delta > thresholds["visual"]:
            mismatch_type.append("visual")
        if txt_delta > thresholds["text"]:
            mismatch_type.append("text")
        if pos_delta > thresholds["spatial"]:
            mismatch_type.append("position")
        if class_changed:
            mismatch_type.append("class")

        match["mismatch_type"] = mismatch_type
        match["delta_scores"] = {
            "visual": round(vis_delta, 4),
            "text": round(txt_delta, 4),
            "position": pos_delta,
            "class_changed": class_changed
        }

    return results

def strip_all_embeddings(results):
    """
    Remove embedding keys to make results JSON-light and shareable.
    """

    def clean_component(comp):
        comp = comp.copy()
        for key in [
            "embedding_visual",
            "embedding_text",
            "embedding_spatial",
            "embedding_class",
            "embedding_fused"
        ]:
            comp.pop(key, None)
        return comp

    def clean_match(match):
        return {
            key: clean_component(value) if key in ["golden", "actual"] else value
            for key, value in match.items()
        }

    results["matches"] = [clean_match(m) for m in results.get("matches", [])]
    results["unmatched_golden"] = [clean_component(c) for c in results.get("unmatched_golden", [])]
    results["unmatched_actual"] = [clean_component(c) for c in results.get("unmatched_actual", [])]

    return results
