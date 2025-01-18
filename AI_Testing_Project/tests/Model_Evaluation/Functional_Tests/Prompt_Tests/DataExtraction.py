#Example: Structured data extraction

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message_data_extraction = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Extract the name, size, price, and color from this product description as a JSON object: " +
        "<description> " +
        "The SmartHome Mini is a compact smart home assistant available in black or white for only $49.99. " +
        "At just 5 inches wide, it lets you control lights, thermostats, and other connected devices via voice or " +
        "app—no matter where you place it in your home. This affordable little hub brings convenient hands-free control to your smart devices. " +
        "</description>"
        }     
        ]
)
print(message_data_extraction.content)