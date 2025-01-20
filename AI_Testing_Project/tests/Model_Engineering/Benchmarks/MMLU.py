import json
import pandas as pd
from anthropic import Anthropic
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import os
from pathlib import Path
import sys

# Add config directory to path
config_path = Path(__file__).parent.parent.parent / 'config'
sys.path.append(str(config_path))

from config import *

def load_mmlu_data(path: str) -> pd.DataFrame:
    """
    Load MMLU dataset from a CSV file.
    Expected format: question, A, B, C, D, answer
    """
    df = pd.read_csv(f"{Config.dataset_folder}/mmlu_sample_data.csv")
    return df

def format_mmlu_prompt(row: pd.Series) -> str:
    """
    Format a single MMLU question into a prompt for Claude.
    """
    prompt = f"""Question: {row['question']}

A) {row['A']}
B) {row['B']}
C) {row['C']}
D) {row['D']}

Please choose A, B, C, or D. Respond with just the letter of your answer."""
    return prompt

def evaluate_mmlu(
    api_key: str,
    data_path: str,
    batch_size: int = 50,
    model: str = "claude-3-5-sonnet-20241022"
) -> Dict:
    """
    Evaluate Claude model on MMLU benchmark.
    
    Args:
        api_key: Anthropic API key
        data_path: Path to MMLU CSV dataset
        batch_size: Number of questions to evaluate at once
        model: Model identifier to use
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)
    
    # Load data
    df = load_mmlu_data(data_path)
    
    # Store results
    correct = 0
    total = 0
    predictions = []
    
    # Process in batches
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            prompt = format_mmlu_prompt(row)
            
            try:
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
                pred = response.content[0].text.strip().upper()
                predictions.append(pred)
                
                # Check if correct
                if pred == row['answer'].strip().upper():
                    correct += 1
                total += 1
                
            except Exception as e:
                print(f"Error processing question: {e}")
                predictions.append("ERROR")
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    results = {
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": accuracy,
        "predictions": predictions
    }
    
    return results

def main():
    # Configuration
    API_KEY = os.getenv('$ANTHROPIC_API_KEY')
    DATA_PATH = "C:\\Users\\SandraDujmovic\\Documents\\AI_Testing_Projects\\ClaudeAI\\tests\\LLM\\Benchmarks\\Dataset\\mmlu_sample_data.csv"
    
    # Run evaluation
    results = evaluate_mmlu(API_KEY, DATA_PATH)
    
    # Print results
    print("\nMMLU Evaluation Results:")
    print(f"Total Questions: {results['total_questions']}")
    print(f"Correct Answers: {results['correct_answers']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    
    # Save results
    with open("mmlu_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
