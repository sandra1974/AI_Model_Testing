#Example: If a user has been discussing a trip to Paris for a
#while, and later asks, "What are some great restaurants
#there?", the model should understand that "there" refers to
#Paris and not need clarification.

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is the capital of France"},
         {"role": "user", "content": "What are some great restaurants there?"}
    ]
)
print(message.content)

