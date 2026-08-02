import base64
import os
import time
from datetime import datetime

from openai import OpenAI

from agents.base_agent import BaseAgent
from config import OPENAI_API_KEY, OUTPUT_IMAGES
from logger import logger

PROMPT_STYLES = {
    "realistic": """
Photorealistic
Ultra detailed
Natural lighting
8K
""",

    "anime": """
Anime illustration
Studio Ghibli style
Soft colors
Detailed shading
""",

    "cyberpunk": """
Cyberpunk city
Neon lights
Rain
Ultra detailed
""",

    "watercolor": """
Watercolor painting
Soft brush strokes
Pastel colors
"""
}


def detect_style(query: str) -> str:
    text = query.lower()

    if any(word in text for word in [
        "anime", "manga", "ghibli", "cartoon"
    ]):
        return "anime"

    if any(word in text for word in [
        "cyberpunk", "neon", "futuristic", "sci-fi"
    ]):
        return "cyberpunk"

    if any(word in text for word in [
        "watercolor", "painting", "paint", "brush"
    ]):
        return "watercolor"

    return "realistic"


class ImageAgent(BaseAgent):

    def __init__(self):
        self.client = None

    def handle(self, query):
        for attempt in range(3):

            try:

                if self.client is None:

                    if not OPENAI_API_KEY:
                        return "❌ OpenAI API key is not configured."

                    self.client = OpenAI(api_key=OPENAI_API_KEY)

                os.makedirs(OUTPUT_IMAGES, exist_ok=True)
                style = detect_style(query)
                logger.info(f"Selected image style: {style}")
                style_prompt = PROMPT_STYLES.get(style, PROMPT_STYLES["realistic"])
                # Prompt Engineering
                enhanced_prompt = f"""
                You are an expert AI image generator.
    
                Style:
                {style_prompt}
    
                Requirements:
                - High quality
                - Professional composition
                - Sharp details
                - Realistic lighting
                - Clean background unless requested otherwise
    
                User Request:
                {query}
                """
                logger.info(f"Enhanced Prompt:\n{enhanced_prompt}")
                result = self.client.images.generate(
                    model="gpt-image-1",
                    prompt=enhanced_prompt,
                    size="1024x1024",
                    quality="high"
                )

                image_bytes = base64.b64decode(result.data[0].b64_json)

                filename = (
                    f"{style}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                filepath = os.path.join(OUTPUT_IMAGES, filename)

                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                return (
                    "✅ Image generated successfully.\n\n"
                    f"Saved at:\n{filepath}"
                )

            except Exception as e:

                logger.exception("ImageAgent failed")

                message = str(e).lower()

                if "insufficient_quota" in message:
                    return (
                        "❌ Your OpenAI image generation quota has been exceeded.\n\n"
                        "Please check your OpenAI billing account."
                    )

                if "invalid_api_key" in message:
                    return (
                        "❌ Invalid OpenAI API key."
                    )

                if attempt == 2:
                    break

                logger.warning(f"Retrying image generation ({attempt + 1}/3)...")
                time.sleep(2 ** attempt)

        return (
            "❌ Unable to generate the image at the moment.\n\n"
            "Please try again later."
        )
