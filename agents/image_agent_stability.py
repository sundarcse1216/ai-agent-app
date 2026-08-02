import os
from datetime import datetime

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from stability_sdk import client

from agents.base_agent import BaseAgent
from config import STABILITY_API_KEY, OUTPUT_IMAGES, IMAGE_MODEL


class ImageAgent(BaseAgent):

    def __init__(self):
        self.stability_api = None

    def handle(self, query):

        try:

            if self.stability_api is None:

                if not STABILITY_API_KEY:
                    return "Stability AI API key is not configured."

                self.stability_api = client.StabilityInference(
                    key=STABILITY_API_KEY,
                    engine=IMAGE_MODEL
                )

            os.makedirs(OUTPUT_IMAGES, exist_ok=True)

            answers = self.stability_api.generate(
                prompt=query,
                steps=30,
                width=1024,
                height=1024
            )

            for response in answers:
                for artifact in response.artifacts:

                    if artifact.finish_reason == generation.FILTER:
                        return "Image generation failed due to safety filter."

                    filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

                    filepath = os.path.join(OUTPUT_IMAGES, filename)

                    with open(filepath, "wb") as f:
                        f.write(artifact.binary)

                    return f"Image generated successfully.\nSaved at: {filepath}"

            return "Unable to generate image."

        except Exception as e:
            return f"Image generation failed: {e}"
