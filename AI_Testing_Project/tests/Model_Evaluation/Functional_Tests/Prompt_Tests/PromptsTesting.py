#Asking a text generation model, "What is the capital of France?" without
#having trained the model specifically on geography to evaluate how well it
#generalizes knowledge.

import anthropic
from anthropic import HUMAN_PROMPT, AI_PROMPT
import base64
import httpx
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message_basic = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hi, please translate from English to Spanish this text \'LLM is a computer program that has been fed enough examples to be able to recognize and interpret human language or other types of complex data. Many LLMs are trained on data that has been gathered from the Internet — thousands or millions of gigabytes' worth of text. But the quality of the samples impacts how well LLMs will learn natural language, so an LLM's programmers may use a more curated data set.\'"}
    ]
)
print(message_basic.content)

message_simple = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
print(message_simple.content)

message_multiple_turns = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Can you describe LLMs to me?"}
    ],
)
print(message_multiple_turns)

message_pre_fill = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1,
    messages=[
        {"role": "user", "content": "What is the official language in Brazil? (A) Spanish, (B) Portuguese, (C) English"},
        {"role": "assistant", "content": "The answer is ("}
    ]
)
print(message_pre_fill)

image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
image_media_type = "image/jpeg"
image_data = base64.standard_b64encode(httpx.get(image_url).content).decode("utf-8")

image_vision = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_data,
                    },
                }
            ],
        }
    ],
)
print(image_vision)

completion = client.completions.create(
    model="claude-2.1",
    max_tokens_to_sample=1024,
    prompt=f"{HUMAN_PROMPT} Hello, Claude{AI_PROMPT}",
)
print(completion)

streaming_completion = client.completions.create(
    model="claude-2",
    max_tokens_to_sample=256,
    prompt="\n\nHuman: Hello, world!\n\nAssistant:",
    stream= True
)
print(streaming_completion)

message_unclear_prompt = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    FEEDBACK_DATA= "I'm not satisfied with your services, Sam Moore"
    messages=[
        {"role": "user", "content": "Please remove all personally identifiable information from these customer feedback messages: {{FEEDBACK_DATA}}"}
    ]
)
print(message_unclear_prompt.content)