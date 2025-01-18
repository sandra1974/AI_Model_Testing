#Example: "If a train leaves at 3 PM and travels for 2 hours at 60 km/h, how far
#will it have traveled by 5 PM?" Here, the model needs to demonstrate logical
#reasoning by calculating the total distance based on the input information.

#Example: Financial Analysis - without thinking

#Example: Financial Analysis - with thinking

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "If a train leaves at 3 PM and travels for 2 hours at 60 km/h, how far will it have traveled by 5 PM?"}
    ]
)
print(message.content)

message_financial_analysis_without_thinking = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "You’re a financial advisor. A client wants to invest $10,000. " +
                                    "They can choose between two options: " +
                                    "A) A stock that historically returns 12% annually but is volatile, or " +
                                    "B) A bond that guarantees 6% annually. " +
                                    "The client needs the money in 5 years for a down payment on a house. Which option do you recommend?"}
    ]
)
print(message_financial_analysis_without_thinking.content)

message_financial_analysis_with_thinking = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "You’re a financial advisor. A client wants to invest $10,000. " +
                                    "They can choose between two options: " +
                                    "A) A stock that historically returns 12% annually but is volatile, or " +
                                    "B) A bond that guarantees 6% annually. " +
                                    "The client needs the money in 5 years for a down payment on a house. Which option do you recommend? Think step-by-step."}
    ]
)
print(message_financial_analysis_with_thinking.content)