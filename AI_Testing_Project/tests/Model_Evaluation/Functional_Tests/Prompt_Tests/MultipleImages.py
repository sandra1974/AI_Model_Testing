import anthropic
from anthropic import HUMAN_PROMPT, AI_PROMPT
import base64
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

# Add config directory to path
config_path = Path(__file__).parent.parent / 'config'
sys.path.append(str(config_path))

from config import * 

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

with open(f"{Config.image_folder}/image01.jpeg", "rb") as f:

    image1_data = base64.b64encode(f.read()).decode("utf-8")
    image1_media_type = "image/jpeg"

    with open(f"{Config.image_folder}/image02.jpeg", "rb") as fr:
        image2_data = base64.b64encode(fr.read()).decode("utf-8")   
        image2_media_type = "image/jpeg"

image_vision = client.messages.create(
     model="claude-3-5-sonnet-20241022",
     system="Respond only in Croatian.",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                 {
                    "type": "text",
                    "text": "Image 1:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image1_media_type,
                        "data": image1_data
                    },                   
                },
                 {
                    "type": "text",
                    "text": "Image 2:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image2_media_type,
                        "data": image2_data,
                    },
                },
                {
                    "type": "text",
                    "text": "How are these images different?"
                }
            ],
        }
    ],
)
print(image_vision)