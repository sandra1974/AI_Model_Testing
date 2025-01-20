import json
import pandas as pd
from anthropic import Anthropic
from tqdm import tqdm
import numpy as np
from typing import Dict, List
import os
from pathlib import Path
import sys

# Add config directory to path
config_path = Path(__file__).parent / 'config'
sys.path.append(str(config_path))

def load_hellaswag_data(path: str) -> pd.DataFrame:
    """
    Load HellaSwag dataset from a JSON file.
    Expected format includes: context/activity_label, endings (4 choices), label (correct answer index)
    """
    try:
        with open(f"{Config.dataset_folder}/hellaswag_sample_data.json", "r") as f:
            data = json.load(f)
        
        # Convert to DataFrame if it's a list of dictionaries
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Data should be a list of dictionaries")
        
        # Print data structure for debugging
        print("\nData structure of first item:")
        print(json.dumps(data[0], indent=2))
        print("\nAvailable columns:", df.columns.tolist())
        
        required_fields = {
            'ctx': ['ctx', 'context', 'activity_label', 'text'],
            'endings': ['endings', 'choices', 'options'],
            'label': ['label', 'answer', 'correct_idx']
        }
        
        # Map fields to standard names
        final_columns = {}
        for required, alternatives in required_fields.items():
            found = False
            for alt in alternatives:
                if alt in df.columns:
                    final_columns[alt] = required
                    found = True
                    break
            if not found:
                raise KeyError(f"Could not find field '{required}'. Alternatives: {alternatives}")
        
        df = df.rename(columns=final_columns)
        return df
        
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        print("\nExpected JSON structure:")
        print("""
        [
            {
                "ctx": "context or activity description",
                "endings": ["ending1", "ending2", "ending3", "ending4"],
                "label": 0  // index of correct ending
            },
            ...
        ]
        """)
        raise

def format_hellaswag_prompt(context: str, endings: List[str]) -> str:
    """
    Format a HellaSwag question into a prompt for Claude.
    """
    prompt = f"""Complete the following scenario by choosing the most appropriate ending. 
    
Context: {context}

Possible endings:
A) {endings[0]}
B) {endings[1]}
C) {endings[2]}
D) {endings[3]}

Please respond with just the letter (A, B, C, or D) of the most natural and likely ending."""
    return prompt

def index_to_letter(index: int) -> str:
    """Convert numeric index to letter answer."""
    return chr(65 + index)  # 0 -> A, 1 -> B, etc.

def letter_to_index(letter: str) -> int:
    """Convert letter answer to numeric index."""
    return ord(letter.upper()) - 65  # A -> 0, B -> 1, etc.

def evaluate_hellaswag(
    api_key: str,
    data_path: str,
    batch_size: int = 50,
    model: str = "claude-3-5-sonnet-20241022"
) -> Dict:
    """
    Evaluate Claude model on HellaSwag benchmark.
    
    Args:
        api_key: Anthropic API key
        data_path: Path to HellaSwag dataset
        batch_size: Number of questions to evaluate at once
        model: Model identifier to use
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)
    
    # Load data
    print("Loading HellaSwag dataset...")
    df = load_hellaswag_data(data_path)
    print(f"Loaded {len(df)} questions successfully.")
    
    # Store results
    results = []
    correct = 0
    total = 0
    
    # Process in batches
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            try:
                # Format prompt
                prompt = format_hellaswag_prompt(row['ctx'], row['endings'])
                
                # Get model's response
                response = client.messages.create(
                    model=model,
                    max_tokens=1,
                    temperature=0,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                
                # Extract prediction (assuming Claude responds with just the letter)
                pred_letter = response.content[0].text.strip().upper()
                pred_index = letter_to_index(pred_letter)
                
                # Get correct answer
                correct_index = int(row['label'])
                correct_letter = index_to_letter(correct_index)
                
                # Check if correct
                is_correct = pred_index == correct_index
                if is_correct:
                    correct += 1
                total += 1
                
                results.append({
                    'context': row['ctx'],
                    'correct_ending': row['endings'][correct_index],
                    'predicted_ending': row['endings'][pred_index],
                    'correct_answer': correct_letter,
                    'predicted_answer': pred_letter,
                    'is_correct': is_correct
                })
                
            except Exception as e:
                print(f"\nError processing question: {str(e)}")
                print(f"Question data: {row.to_dict()}")
                results.append({
                    'context': row['ctx'] if 'ctx' in row else 'ERROR',
                    'correct_ending': 'ERROR',
                    'predicted_ending': 'ERROR',
                    'correct_answer': 'ERROR',
                    'predicted_answer': 'ERROR',
                    'is_correct': False
                })
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    evaluation_results = {
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": accuracy,
        "detailed_results": results
    }
    
    return evaluation_results

def main():
    # Configuration
    API_KEY = os.getenv('$ANTHROPIC_API_KEY')
    DATA_PATH = "C:\\Users\\SandraDujmovic\\Documents\\AI_Testing_Projects\\ClaudeAI\\tests\\LLM\\Benchmarks\\Dataset\\hellaswag_sample_data.json"
    
    # Run evaluation
    results = evaluate_hellaswag(API_KEY, DATA_PATH)
    
    # Print results
    print("\nHellaSwag Evaluation Results:")
    print(f"Total Questions: {results['total_questions']}")
    print(f"Correct Answers: {results['correct_answers']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    
    # Save results
    with open("hellaswag_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
