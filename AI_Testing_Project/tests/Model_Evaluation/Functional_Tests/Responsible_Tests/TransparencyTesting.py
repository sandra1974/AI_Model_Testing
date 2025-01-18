#Example: Loan approval model explaining reasons for approval or denial

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message_transparency_prompt = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain the reasons why loan can be approved or denied. "}
             ]
)
print(message_transparency_prompt.content)