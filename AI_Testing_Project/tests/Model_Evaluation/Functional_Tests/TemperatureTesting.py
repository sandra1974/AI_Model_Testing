#Example: For a chatbot that generates responses, setting the temperature to
#0.2 will make the responses more repetitive and focused, whereas a
#temperature of 0.9 will make the responses more varied and creative. Testing
#different temperatures can help evaluate the model's stability.

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message_multiple_turns = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    temperature=0.9,
    messages=[
        {"role": "user", "content": "Hello, Claude"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Can you describe LLMs to me?"}
    ],
)
print(message_multiple_turns)