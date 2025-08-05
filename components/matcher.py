import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from difflib import SequenceMatcher


def compute_cosine_similarity(vec1, vec2):
    return cosine_similarity(np.array(vec1).reshape(1, -1), np.array(vec2).reshape(1, -1))[0][0]


def bbox_center_distance(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
    cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
    return np.linalg.norm([cx1 - cx2, cy1 - cy2])


def bbox_iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    b1_area = w1 * h1
    b2_area = w2 * h2

    union_area = b1_area + b2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def text_similarity(a, b):
    """Compute normalized text similarity between two strings."""
    a = str(a).lower() if isinstance(a, str) else ""
    b = str(b).lower() if isinstance(b, str) else ""
    return SequenceMatcher(None, a, b).ratio()



def hungarian_match_components(golden, actual, match_threshold=0.85):
    n, m = len(golden), len(actual)
    cost_matrix = np.ones((n, m))

    for i, g in enumerate(golden):
        for j, a in enumerate(actual):
            if g.get("embedding_fused") is None or a.get("embedding_fused") is None:
                continue

            emb_sim = compute_cosine_similarity(g["embedding_fused"], a["embedding_fused"])
            iou = bbox_iou(g["bbox"], a["bbox"])
            text_sim = text_similarity(g.get("text", ""), a.get("text", ""))

            # Add penalty if component types differ
            type_penalty = 0.1 if g.get("type") != a.get("type") else 0

            # Weighted matching score
            score = (
                0.6 * emb_sim +
                0.2 * iou +
                0.2 * text_sim -
                type_penalty
            )

            # Clamp score between 0 and 1
            score = max(0, min(score, 1.0))
            cost_matrix[i, j] = 1 - score

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_golden = set(range(n))
    unmatched_actual = set(range(m))

    for i, j in zip(row_ind, col_ind):
        score = 1 - cost_matrix[i, j]
        if score >= match_threshold:
            matches.append({
                "golden": golden[i],
                "actual": actual[j],
                "match_score": round(score, 4),
                "embedding_similarity": round(compute_cosine_similarity(
                    golden[i]["embedding_fused"], actual[j]["embedding_fused"]), 4),
                "bbox_iou": round(bbox_iou(golden[i]["bbox"], actual[j]["bbox"]), 4),
                "text_similarity": round(text_similarity(
                    golden[i].get("text", ""), actual[j].get("text", "")), 4),
                "position_distance": round(bbox_center_distance(golden[i]["bbox"], actual[j]["bbox"]), 2)
            })
            unmatched_golden.discard(i)
            unmatched_actual.discard(j)

    return {
        "matches": matches,
        "unmatched_golden": [golden[i] for i in unmatched_golden],
        "unmatched_actual": [actual[j] for j in unmatched_actual]
    }
