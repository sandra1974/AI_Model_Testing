#Example: When asked, "Explain the causes of global warming," the model should
#stick to discussing climate change rather than suddenly switching topics
#to pollution control unless it's closely related.

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain the causes of global warming"}
    ]
)
print(message)