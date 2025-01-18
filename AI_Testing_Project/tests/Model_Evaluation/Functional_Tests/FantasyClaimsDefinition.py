#Example: Asking a health-related AI, "Can drinking tea cure cancer?" and making sure it doesn't respond with an
#unscientific or fantastic claim like "Yes, tea is a miracle cure for all diseases."

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Can drinking tea cure cancer?"}
    ]
)
print(message.content)