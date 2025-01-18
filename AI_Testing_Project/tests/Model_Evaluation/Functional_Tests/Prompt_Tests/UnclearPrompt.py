#Example: Anonymizing cusstomer feedback - Unclear prompt

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message_unclear_prompt = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Please remove all personally identifiable information from these customer feedback messages: \'I am not satisfied with your services, Sandra Rozmanic\'"}
    ]
)
print(message_unclear_prompt.content)