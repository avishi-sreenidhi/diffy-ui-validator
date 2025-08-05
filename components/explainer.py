import json
import cv2
import base64
import os
from openai import OpenAI


def encode_cv2_image(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def simplify_entry(entry):
    # Keep only relevant fields for LLM
    allowed_keys = ['text', 'component', 'mismatch_type', 'delta', 'expected', 'actual', 'bbox']
    return {k: v for k, v in entry.items() if k in allowed_keys}


def explain_ui_mismatches(
    semantic_meaning: str,
    results: dict,
    golden_img=None,
    actual_img=None,
    model_id: str = "gpt-4o"
) -> str:
    """
    Uses GPT-4o to generate developer-friendly explanations of UI mismatches
    with pixel-level and color-specific suggestions.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Extract core fields
    theme_flag = results.get("overall_theme_issue", "No issue")
    matches = results.get("matches", [])
    unmatched_golden = results.get("unmatched_golden", [])
    unmatched_actual = results.get("unmatched_actual", [])

    # Trim lists to avoid token overload
    simplified_matches = [simplify_entry(m) for m in matches if m.get("mismatch_type")][:15]
    simplified_unmatched_golden = [simplify_entry(u) for u in unmatched_golden][:10]
    simplified_unmatched_actual = [simplify_entry(u) for u in unmatched_actual][:10]

    messages = []

    # --- System Prompt ---
    system_prompt = """
You are a UI regression validation assistant helping QA engineers and frontend developers fix mismatches.

You are given:
- A screen purpose
- Theme flag (dark/light mismatch, etc.)
- UI metadata (component type, text, bounding boxes, colors, class, etc.)
- Optional screenshots

Your job is to:
1. **Theme Verification**: Validate if theme differences are real using images and metadata. Mention only if relevant.
2. **Mismatch Detection**: Confirm mismatches using text, bbox, visual cues, and color. Use screenshot only if metadata is ambiguous.
3. **Clarity**: Explain mismatches clearly (e.g., *Text changed from “COLD BREWS” to “COFFEE BREWS”* or *Button shifted right by 42px*).
4. **Missing Components**: If a component from the design is missing in render (especially text content), flag it clearly.
5. **Developer-Focused Fix Suggestions**: Provide **explicit, actionable suggestions**, e.g.:
    - “Shift button 24px to the left (from x=320 to x=296)”
    - “Change text color from `#F5F5F5` to `#000000`”
    - “Add missing text block: ‘Your perfect drink is just a few clicks away’ at y=1600”

Output Format:
- Verified Mismatches
- Missing Components
- Extra Components
- Suggested Fixes
"""
    messages.append({"role": "system", "content": system_prompt.strip()})

    # --- User Prompt ---
    user_prompt = f"""
Screen Purpose: {semantic_meaning}
Theme Flag: {theme_flag}

Mismatched Components:
{json.dumps(simplified_matches, indent=2)}

Missing from Rendered UI:
{json.dumps(simplified_unmatched_golden, indent=2)}

Extra in Rendered UI:
{json.dumps(simplified_unmatched_actual, indent=2)}

Pay special attention to textual content, bounding box coordinates (x, y, width, height), and color values (if available in `expected` / `actual`). Your suggestions should be helpful to frontend developers aiming for pixel-perfect design.
"""
    messages.append({"role": "user", "content": user_prompt.strip()})

    # --- Image Messages (if available) ---
    if golden_img is not None and actual_img is not None:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Baseline (design) screenshot:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_cv2_image(golden_img)}"}}
            ]
        })
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Rendered (actual) UI screenshot:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_cv2_image(actual_img)}"}}
            ]
        })

    # --- LLM Completion ---
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"LLM explanation failed: {str(e)}")
