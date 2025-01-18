import os
from datasets import load_dataset
from anthropic import Anthropic
import pandas as pd
from tqdm import tqdm

def evaluate_mmlu(api_key, subset_name=None, num_samples=None):
    """
    Evaluate Claude on MMLU dataset
    
    Args:
        api_key (str): Anthropic API key
        subset_name (str, optional): Specific MMLU subset to evaluate
        num_samples (int, optional): Number of samples to evaluate
    """
    # Initialize Anthropic client
    client = Anthropic(api_key=os.getenv('$ANTHROPIC_API_KEY'))
    
    # Load MMLU dataset
    dataset = load_dataset("cais/mmlu", subset_name if subset_name else "all")["test"]
    
    if num_samples:
        dataset = dataset.select(range(num_samples))
    
    results = []
    
    for item in tqdm(dataset):
        # Format the question with multiple choice options
        prompt = f"""Question: {item['question']}

A) {item['choices'][0]}
B) {item['choices'][1]}
C) {item['choices'][2]}
D) {item['choices'][3]}

Please respond with just a single letter (A, B, C, or D) corresponding to the correct answer."""

        # Get model's response
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1
        )
        
        # Extract model's answer
        model_answer = response.content[0].text.strip()
        
        # Convert numeric answer to letter (MMLU uses 0-3 for answers)
        correct_answer = chr(ord('A') + item['answer'])
        
        # Store results
        results.append({
            'subject': item['subject'],
            'question': item['question'],
            'model_answer': model_answer,
            'correct_answer': correct_answer,
            'is_correct': model_answer == correct_answer
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate overall accuracy
    overall_accuracy = results_df['is_correct'].mean()
    
    # Calculate per-subject accuracy
    subject_accuracy = results_df.groupby('subject')['is_correct'].mean()
    
    return {
        'overall_accuracy': overall_accuracy,
        'subject_accuracy': subject_accuracy,
        'detailed_results': results_df
    }

if __name__ == "__main__":
    # Replace with your API key
    API_KEY = os.getenv('$ANTHROPIC_API_KEY')
    
    # Example usage
    results = evaluate_mmlu(
        api_key=API_KEY,
        subset_name="abstract_algebra",  # Optional: specify subset
        num_samples=10  # Optional: limit number of samples
    )
    
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.2%}")
    print("\nAccuracy by Subject:")
    print(results['subject_accuracy'])