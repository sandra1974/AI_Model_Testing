#Example: Analyzing customer feedback - with and without example

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message_prompt_no_example = client.messages.create(
   model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Analyze this customer feedback and categorize the issues. "+
       "Use these categories: UI/UX, Performance, Feature Request, Integration, Pricing, and Other. " +
        "Also rate the sentiment (Positive/Neutral/Negative) and priority (High/Medium/Low). " +
        "Here is the feedback: " +
        "\'Total button which appears on Cart screen contains colon what is not correct. Colon should be removed. Medium Priority.\'"                  
        }        
            ]
)
print(message_prompt_no_example.content)


message_prompt_with_example = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Our CS team is overwhelmed with unstructured feedback. " +
        "Your task is to analyze feedback and categorize issues for our product and engineering teams. " +
        "Use these categories: UI/UX, Performance, Feature Request, Integration, Pricing, and Other. " +
        "Also rate the sentiment (Positive/Neutral/Negative) and priority (High/Medium/Low). Here is an example: " +
        "<example>Input: The new dashboard is a mess! It takes forever to load, and I can’t find the export button. Fix this ASAP!Category: UI/UX, PerformanceSentiment: NegativePriority: High</example> " +
        "Now, analyze this feedback: " +
        "\'Total button which appears on Cart screen contains colon what is not correct. Colon should be removed. Medium Priority.\'"  
                   
        }        
            ]
)
print(message_prompt_with_example.content)