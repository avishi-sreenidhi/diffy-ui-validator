import numpy as np
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

# ---- Utility functions ----

def compute_cosine_similarity(vec1, vec2):
    return cosine_similarity(np.array(vec1).reshape(1, -1), np.array(vec2).reshape(1, -1))[0][0]

def bbox_center_distance(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
    cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
    return np.linalg.norm([cx1 - cx2, cy1 - cy2])

def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    union = float(boxAArea + boxBArea - interArea)
    return interArea / union if union != 0 else 0

def text_similarity(t1, t2):
    if not isinstance(t1, str):
        t1 = ""
    if not isinstance(t2, str):
        t2 = ""
    return difflib.SequenceMatcher(None, t1.lower(), t2.lower()).ratio()

def type_penalty(type1, type2):
    return 0 if type1 == type2 else -0.1

def is_contained(inner, outer):
    x1, y1, w1, h1 = inner
    x2, y2, w2, h2 = outer
    return (x1 >= x2 and y1 >= y2 and
            x1 + w1 <= x2 + w2 and
            y1 + h1 <= y2 + h2)

def normalize_bbox(bbox, precision=10):
    """Round bbox coordinates to nearest multiple of `precision`."""
    return [round(x / precision) * precision for x in bbox]

# ---- Hierarchical Grouping ----

def group_components_hierarchically(components):
    grouped = []
    used_children = set()

    # Normalize bboxes first
    for comp in components:
        comp["bbox"] = normalize_bbox(comp["bbox"])

    for i, parent in enumerate(components):
        parent_bbox = parent["bbox"]
        parent_children = []

        for j, child in enumerate(components):
            if i == j or j in used_children:
                continue
            child_bbox = child["bbox"]
            if is_contained(child_bbox, parent_bbox):
                parent_children.append(child)
                used_children.add(j)

        if parent_children:
            parent["children"] = parent_children

            # Merge child texts
            child_texts = " ".join([str(c.get("text", "") or "") for c in parent_children])
            parent_text = str(parent.get("text") or "")
            parent["text"] = parent_text + " " + child_texts

            # Merge embeddings
            child_embeddings = [c.get("embedding_fused") for c in parent_children if isinstance(c.get("embedding_fused"), np.ndarray)]
            if parent.get("embedding_fused") is not None and child_embeddings:
                all_embeddings = [parent["embedding_fused"]] + child_embeddings
                parent["embedding_fused"] = np.mean(np.array(all_embeddings), axis=0)

        grouped.append(parent)

    # Add leftover components
    for i, comp in enumerate(components):
        if i not in used_children and comp not in grouped:
            grouped.append(comp)

    return grouped

# ---- Main Matching Function ----

def hungarian_match_components(golden, actual, match_threshold=0.85):
    golden = group_components_hierarchically(golden)
    actual = group_components_hierarchically(actual)

    n, m = len(golden), len(actual)
    cost_matrix = np.ones((n, m))

    for i, g in enumerate(golden):
        for j, a in enumerate(actual):
            if g.get("embedding_fused") is not None and a.get("embedding_fused") is not None:
                emb_sim = compute_cosine_similarity(g["embedding_fused"], a["embedding_fused"])
                txt_sim = text_similarity(g.get("text", ""), a.get("text", ""))
                pos_dist = bbox_center_distance(g["bbox"], a["bbox"])
                iou = bbox_iou(g["bbox"], a["bbox"])
                type_sim = 1 + type_penalty(g.get("type", ""), a.get("type", ""))

                weighted_sim = (
                    0.5 * emb_sim +
                    0.2 * txt_sim +
                    0.2 * iou +
                    0.1 * type_sim
                )

                cost_matrix[i, j] = 1 - weighted_sim

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_golden = set(range(n))
    unmatched_actual = set(range(m))

    for i, j in zip(row_ind, col_ind):
        sim_score = 1 - cost_matrix[i, j]
        if sim_score >= match_threshold:
            matches.append({
                "golden": golden[i],
                "actual": actual[j],
                "embedding_similarity": round(sim_score, 4),
                "position_distance": round(bbox_center_distance(golden[i]["bbox"], actual[j]["bbox"]), 2),
                "iou": round(bbox_iou(golden[i]["bbox"], actual[j]["bbox"]), 4),
                "text_similarity": round(text_similarity(golden[i].get("text", ""), actual[j].get("text", "")), 4)
            })
            unmatched_golden.discard(i)
            unmatched_actual.discard(j)

    return {
        "matches": matches,
        "unmatched_golden": [golden[i] for i in unmatched_golden],
        "unmatched_actual": [actual[j] for j in unmatched_actual]
    }
