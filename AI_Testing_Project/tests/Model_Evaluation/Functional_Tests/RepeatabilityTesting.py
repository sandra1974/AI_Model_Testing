#Example: Asking a model, "What is the square root of 16?", "Calculate square root of 16", "Square root of 16 is ..." multiple times should always yield "4" as the answer. If the
#answer varies, it indicates instability in the model.

#Example: For an AI answering a math question, "What is 5 + 7?" and "Could you add five and seven for me?" should both
#return the answer "12."

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Square root of 16 is ..."}
    ]
)
print(message.content)

message_repeat = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is the square root of 16?"}              
    ]
)
print(message_repeat.content)

message_repeat_again = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[      
         {"role": "user", "content": "Calculate square root of 16"},
         ]
)
print(message_repeat_again.content)