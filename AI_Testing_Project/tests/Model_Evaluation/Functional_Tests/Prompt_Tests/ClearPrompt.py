#Example: Anonymizing cusstomer feedback - Clear prompt

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message_clear_prompt = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Your task is to anonymize customer feedback for our quarterly review. Instructions: " +
                        "1. Replace all customer names with \'CUSTOMER_[ID]\' (e.g., \'Jane Doe\' → \'CUSTOMER_001\'). " +
                        "2. Replace email addresses with \'EMAIL_[ID]@example.com\'. " +
                        "3. Redact phone numbers as \'PHONE_[ID]\'. " +
                        "4. If a message mentions a specific product (e.g., \'AcmeCloud\'), leave it intact. " +
                        "5. If no PII is found, copy the message verbatim. " +
                        "6. Output only the processed messages, separated by \'---\'." +
                        " Data to process: \'I give my feed back here. My name: Sandra Rozmanic. Email: sandrar@gmail.com. Phone: 385 98 906 1212. I have bad experience with product CellOps.\' "}     
            ]
)
print(message_clear_prompt.content)




