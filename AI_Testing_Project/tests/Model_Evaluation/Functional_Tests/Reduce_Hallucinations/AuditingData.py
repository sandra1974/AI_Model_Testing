import anthropic
import base64
import os
from dotenv import load_dotenv
from pathlib import Path
import sys

# Add config directory to path
config_path = Path(__file__).parent.parent / 'config'
sys.path.append(str(config_path))

from config import * 

load_dotenv()

# Read local PDF file
with open(f"{Config.pdf_folder}/PrivacyPolicy.pdf", "rb") as f:
    pdf_data = base64.b64encode(f.read()).decode("utf-8")

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
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_data
                    }
                },
                {
                    "type": "text",
                    "text": "As our Data Protection Officer, review this updated privacy policy for GDPR and CCPA compliance. 1. Extract exact quotes from the policy that are most relevant to GDPR and CCPA compliance. If you can’t find relevant quotes, state “No relevant quotes found. 2. Use the quotes to analyze the compliance of these policy sections, referencing the quotes by number. Only base your analysis on the extracted quotes."                    
                }
            ]
        }
    ]
)

final_message = message.content[0].text
print(final_message)