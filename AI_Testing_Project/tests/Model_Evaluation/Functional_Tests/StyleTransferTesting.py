#Example of Style Transfer: A text generation model could be asked to write a formal email, and
#then a casual message on the same topic. The responses should differ
#in tone, formality, and word choice while remaining coherent.

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2048,
    system="At the end of each message, add a text as \'Please let us know if you need more information\'", # <-- role prompt
    messages=[
        {"role": "user", "content": "Give me tips to manage covid"}
    ]
)

print(response.content)