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
with open(f"{Config.pdf_folder}/SampleReport.pdf", "rb") as f:
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
                    "text": "As our M&A advisor, analyze this report. Focus on financial projections, integration risks, and regulatory hurdles. If you are unsure about any aspect or if the report lacks necessary information, say \'I do not have enough information to confidently assess this.\' "                    
                }
            ]
        }
    ]
)

final_message = message.content[0].text
print(final_message)