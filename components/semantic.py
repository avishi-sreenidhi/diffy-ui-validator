# semantic.py
import base64
import io
import os
from PIL import Image
import numpy as np
import cv2
import torch
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

client = OpenAI(api_key=api_key)

def semantic_summary(image_input, prompt="Describe the screen and the semantic relevance in a few lines"):
    """
    Sends an image to GPT-4o for semantic screen context understanding.
    
    Args:
        image_input: path (str), PIL.Image, np.ndarray (cv2 or torch)
        prompt: str - user prompt to guide GPT-4o

    Returns:
        A short semantic description string.
    """

    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    elif isinstance(image_input, (np.ndarray, torch.Tensor)):
        if isinstance(image_input, torch.Tensor):
            image_input = image_input.cpu().numpy()
        image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("Unsupported image format for gpt4o_semantic_summary")

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "In a line, briefly explain the semantic context of this mobile screen"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{img_str}",
                        "detail": "low"
                    }}
                ]
            }
        ],
        max_tokens=75,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()
