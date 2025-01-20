import anthropic
import base64
import os
from dotenv import load_dotenv
from pathlib import Path
import sys

# Add config directory to path
config_path = Path(__file__).parent.parent.parent.parent / 'config'
sys.path.append(str(config_path))

from config import * 

load_dotenv()

with open(f"{Config.image_folder}/shoes.png", "rb") as f:

    image_data = base64.b64encode(f.read()).decode("utf-8")

# Send API request
client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)
message = client.beta.messages.create(
    model="claude-3-5-sonnet-20241022",
    betas=["pdfs-2024-09-25"],
    max_tokens=1024,
    messages=[
        {
            "role": "user", 
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": "What do you deduce about the owner of this shoe?"
                }
            ]
        }
    ],
)

print(message)
