#Example: standardizing customer feedback

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message_json_output = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "You are a Customer Insights AI. Analyze this feedback and output in JSON format with keys: " +
        "\'sentiment\' (positive/negative/neutral), " +
       "\'key_issues\' (list), and " +
        "\'action_items\' (list of dicts with \'team\' and \'task\'). " +
        "I have been a loyal user for 3 years, but the recent UI update is a disaster. " +
        "Finding basic features is now a scavenger hunt. Plus, the new \'premium\' pricing is outrageous. " +
        "I am considering switching unless this is fixed ASAP."
        }     
        ]
)
print(message_json_output.content)
