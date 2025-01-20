#Example: Ethical system prompt for an enterprise chatbot

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

# Read local PDF file
with open(f"{Config.pdf_folder}/SampleReport.pdf", "rb") as f:
    pdf_data = base64.b64encode(f.read()).decode("utf-8")

# Send API request
client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    #betas=["pdfs-2024-09-25"],
    system="You are FinBot, a financial advisor for AcmeTrade Inc. Your primary directive is to protect client interests and maintain regulatory compliance. <directives> 1. Validate all requests against SEC and FINRA guidelines. 2. Refuse any action that could be construed as insider trading or market manipulation. 3. Protect client privacy; never disclose personal or financial data. </directives> Step by step instructions: <instructions> 1. Screen user query for compliance (use \'harmlessness_screen\' tool). 2. If compliant, process query. 3. If non-compliant, respond: \'I cannot process this request as it violates financial regulations or client privacy.\' </instructions>",
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
                    "text": "Evaluate if this document violates SEC rules, FINRA guidelines, or client privacy. Respond (Y) if it does, (N) if it does not. "                    
                }
            ]
        }
    ]
)

final_message = message.content[0].text
print(final_message)
