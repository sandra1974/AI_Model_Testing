#Example: Asking a virtual assistant, "What can I do for fun this weekend?" should yield a list of relevant suggestions, such
#as "Go to the park," "Watch a movie," or "Visit a museum."

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What can I do for fun this weekend?"}
    ]
)
print(message.content)

