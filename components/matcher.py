# matcher.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment


def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return cosine_similarity(np.array(vec1).reshape(1, -1), np.array(vec2).reshape(1, -1))[0][0]


def bbox_center_distance(bbox1, bbox2):
    """Calculate Euclidean distance between centers of two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
    cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
    return np.linalg.norm([cx1 - cx2, cy1 - cy2])


def hungarian_match_components(golden, actual, match_threshold=0.9):
    """
    Match golden and actual components using Hungarian algorithm
    based on cosine similarity of their fused embeddings.

    Args:
        golden (list): List of golden component dicts.
        actual (list): List of actual component dicts.
        match_threshold (float): Minimum cosine similarity to accept match.

    Returns:
        dict: {
            "matches": list of dicts with matched components and similarity,
            "unmatched_golden": list of golden components with no match,
            "unmatched_actual": list of unmatched actual components
        }
    """
    n, m = len(golden), len(actual)
    cost_matrix = np.ones((n, m))  # default cost = 1 (worst match)

    for i, g in enumerate(golden):
        for j, a in enumerate(actual):
            if g.get("embedding_fused") is not None and a.get("embedding_fused") is not None:
                sim = compute_cosine_similarity(g["embedding_fused"], a["embedding_fused"])
                cost_matrix[i, j] = 1 - sim  # similarity to cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_golden = set(range(n))
    unmatched_actual = set(range(m))

    for i, j in zip(row_ind, col_ind):
        sim = 1 - cost_matrix[i, j]
        if sim >= match_threshold:
            matches.append({
    "golden": golden[i],
    "actual": actual[j],
    "embedding_similarity": round(sim, 4),
    "position_distance": round(bbox_center_distance(golden[i]["bbox"], actual[j]["bbox"]), 2)
})

            unmatched_golden.discard(i)
            unmatched_actual.discard(j)

    return {
        "matches": matches,
        "unmatched_golden": [golden[i] for i in unmatched_golden],
        "unmatched_actual": [actual[j] for j in unmatched_actual]
    }
