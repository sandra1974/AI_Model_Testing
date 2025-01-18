import anthropic
import os

tweets = [
    {"text": "This movie was a total waste of time. 👎", "sentiment": "negative"},
    {"text": "The new album is 🔥! Been on repeat all day.", "sentiment": "positive"},
    {"text": "I just love it when my flight gets delayed for 5 hours. #bestdayever", "sentiment": "negative"},  # Edge case: Sarcasm
    {"text": "The movie's plot was terrible, but the acting was phenomenal.", "sentiment": "mixed"},  # Edge case: Mixed sentiment
]

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

def get_completion(prompt: str):
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=50,
        messages=[
        {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

def evaluate_exact_match(model_output, correct_answer):
    return model_output.strip().lower() == correct_answer.lower()

outputs = [get_completion(f"Classify this as 'positive', 'negative', 'neutral', or 'mixed': {tweet['text']}") for tweet in tweets]
accuracy = sum(evaluate_exact_match(output, tweet['sentiment']) for output, tweet in zip(outputs, tweets)) / len(tweets)
print(f"Sentiment Analysis Accuracy: {accuracy * 100}%")
