import anthropic
import os

inquiries = [
    {"text": "This is the third time you've messed up my order. I want a refund NOW!", "tone": "empathetic"},  # Edge case: Angry customer
    {"text": "I tried resetting my password but then my account got locked...", "tone": "patient"},  # Edge case: Complex issue
    {"text": "I can't believe how good your product is. It's ruined all others for me!", "tone": "professional"},  # Edge case: Compliment as complaint
    # ... 97 more inquiries
]

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

def get_completion(prompt: str):
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        messages=[
        {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

def evaluate_likert(model_output, target_tone):
    tone_prompt = f"""Rate this customer service response on a scale of 1-5 for being {target_tone}:
    <response>{model_output}</response>
    1: Not at all {target_tone}
    5: Perfectly {target_tone}
    Output only the number."""

    # Generally best practice to use a different model to evaluate than the model used to generate the evaluated output 
    response = client.messages.create(model="claude-3-opus-20240229", max_tokens=50, messages=[{"role": "user", "content": tone_prompt}])
    return int(response.content[0].text.strip())

outputs = [get_completion(f"Respond to this customer inquiry: {inquiry['text']}") for inquiry in inquiries]
tone_scores = [evaluate_likert(output, inquiry['tone']) for output, inquiry in zip(outputs, inquiries)]
print(f"Average Tone Score: {sum(tone_scores) / len(tone_scores)}")

