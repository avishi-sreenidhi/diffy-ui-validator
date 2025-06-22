import json
import cv2
import base64
from openai import OpenAI
import os

def explain_ui_mismatches(
    semantic_meaning: str,
    results: dict,
    golden_img=None,
    actual_img=None,
    model_id: str = "gpt-4o"
) -> str:
    """
    Sends UI semantic meaning, mismatch data, and optionally golden/actual screenshots
    as in-memory cv2 images to OpenAI GPT-4o for structured UI mismatch analysis.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    theme_issue = results.get("overall_theme_issue", "No issue")

    # Always send all mismatches; let the LLM verify if theme issue is legit
    mismatched = [m for m in results.get("matches", []) if m.get("mismatch_type")]
    unmatched_golden = results.get("unmatched_golden", [])
    unmatched_actual = results.get("unmatched_actual", [])

    #  Token-safe truncation and simplification logic
    MAX_MISMATCHES = 15
    MAX_UNMATCHED = 10

    def simplify_entry(entry):
        allowed_keys = ['component', 'mismatch_type', 'delta', 'expected', 'actual']
        return {k: v for k, v in entry.items() if k in allowed_keys}

    mismatched = [simplify_entry(m) for m in mismatched[:MAX_MISMATCHES]]
    unmatched_golden = [simplify_entry(u) for u in unmatched_golden[:MAX_UNMATCHED]]
    unmatched_actual = [simplify_entry(u) for u in unmatched_actual[:MAX_UNMATCHED]]

    include_images = (golden_img is not None and actual_img is not None)

    def encode_cv2_image(img):
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')

    image_messages = []
    if include_images:
        golden_b64 = encode_cv2_image(golden_img)
        actual_b64 = encode_cv2_image(actual_img)

        image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is the **golden (design) screenshot**."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{golden_b64}"}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is the **actual (rendered) UI screenshot**."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{actual_b64}"}}
                ]
            }
        ]

    system_prompt = """
You are a UI validation assistant.

You receive:
- The semantic intent of a UI screen
- A dictionary of mismatch metadata (matches, unmatched elements, theme detection)
- Screenshots of the golden design and actual UI

Your responsibilities:

1. **Theme Validation**:
    - If a theme difference flag is set, check screenshots.
    - If valid (e.g., dark vs light mode), mention it and ignore superficial stylistic mismatches caused by theme. Dont proceed to any other component checks.
    - If invalid, treat theme flag as noise.

2. **Mismatch Validation**:
    - For every mismatch in the "matches" list:
        - Describe clearly what needs to be changed (colors, positions, etc.) What hexcode changes, pixel shifts from the heuristics provided

3. **Image-based Verification for Unmatched Elements/Missing/Extra Elements**:
    - For each element in `unmatched_golden` and `unmatched_actual`:
        - Use screenshots to verify whether it is **truly missing or extra**.
         - Only report unmatched elements that are **actually absent/present in the screenshot**.
        - If the element **does appear visually**, it's a **false positive** → Do NOT report it.
       
4. **Output Format**:
    - Group findings concisely and in order. Verify with the images if the issue is really present and use the metadata
    to present accurate reports, proposed solutions and next actions.
         False Positives (if any)
    - Provide solutions to developers and QA testers to assist them concisely. Avoid repeating raw JSON or stating trivial confirmations (like “username field exists” unless it causes layout issues).
"""

    user_prompt = f"""
Screen Purpose: {semantic_meaning}

Theme Flag: {theme_issue}

Mismatched Components:
{json.dumps(mismatched, indent=2)}

Unmatched Golden Elements (design says these exist, actual might not):
{json.dumps(unmatched_golden, indent=2)}

Unmatched Actual Elements (actual UI has extra components not in design):
{json.dumps(unmatched_actual, indent=2)}

Your Tasks:
- Use screenshots to validate theme flag.
- For unmatched components, visually confirm presence/absence this is very important. Ignore false positives.
- Categorize and describe real mismatches with clear recommendations.
- Where heuristics (pixel positions or deltas) are provided in the mismatch metadata, use them to report precise positional or size changes, color etc
"""

    try:
        messages = [{"role": "system", "content": system_prompt.strip()}]
        messages.append({"role": "user", "content": user_prompt.strip()})
        if include_images:
            messages.extend(image_messages)

        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}")
