#Example: If a user says, "Oh great, another meeting!" sarcastically, the model should understand that the user likely
#feels negatively about the meeting instead of interpreting it as a positive sentiment.

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Oh great, another meeting!"}
    ]
)
print(message.content)