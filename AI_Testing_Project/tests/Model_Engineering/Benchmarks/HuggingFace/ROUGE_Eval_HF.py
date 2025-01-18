from datasets import load_dataset
import anthropic
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from rouge_score import rouge_scorer
import numpy as np
import os

def load_evaluation_dataset():
    """
    Load dataset from HuggingFace for summarization evaluation
    Using CNN/DailyMail as an example dataset
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation")
    return dataset

def get_model_response(client, article):
    """
    Get summary from Claude 3.5 Sonnet
    """
    prompt = f"""Please provide a concise summary of the following article. 
    Be factual and objective:

    {article}
    """
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        temperature=0,  # Using temperature 0 for consistent outputs
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return str(message.content)

def calculate_rouge_scores(prediction, reference):
    """
    Calculate ROUGE scores for a prediction against a reference
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(prediction, reference)
    
    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_f': scores['rougeL'].fmeasure,
        'rouge1_p': scores['rouge1'].precision,
        'rouge2_p': scores['rouge2'].precision,
        'rougeL_p': scores['rougeL'].precision,
        'rouge1_r': scores['rouge1'].recall,
        'rouge2_r': scores['rouge2'].recall,
        'rougeL_r': scores['rougeL'].recall
    }

def serialize_result(result):
    """
    Convert result dictionary to JSON-serializable format
    """
    return {
        "article": str(result["article"]),
        "reference_summary": str(result["reference_summary"]),
        "model_summary": str(result["model_summary"]),
        "rouge_scores": {k: float(v) for k, v in result["rouge_scores"].items()}
    }

def main():
    # Initialize Anthropic client
    client = anthropic.Client(api_key=os.getenv('$ANTHROPIC_API_KEY'))
    
    # Load dataset
    dataset = load_evaluation_dataset()
    
    # Initialize results storage
    results = []
    
    # Create timestamp for the evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Number of examples to evaluate (adjust as needed)
    num_examples = 100
    
    # Evaluate examples
    for idx in tqdm(range(min(num_examples, len(dataset)))):
        try:
            # Get article and reference summary
            article = dataset[idx]["article"]
            reference_summary = dataset[idx]["highlights"]
            
            # Get model's summary
            model_summary = get_model_response(client, article)
            
            # Calculate ROUGE scores
            rouge_scores = calculate_rouge_scores(model_summary, reference_summary)
            
            # Store results
            result = {
                "article": article,
                "reference_summary": reference_summary,
                "model_summary": model_summary,
                "rouge_scores": rouge_scores
            }
            results.append(serialize_result(result))
            
        except Exception as e:
            print(f"Error processing example {idx}: {str(e)}")
            continue
    
    # Calculate average ROUGE scores
    avg_scores = {
        metric: np.mean([r["rouge_scores"][metric] for r in results])
        for metric in results[0]["rouge_scores"].keys()
    }
    
    # Create summary
    summary = {
        "model": "claude-3-5-sonnet-20241022",
        "total_examples": len(results),
        "average_rouge_scores": avg_scores,
        "timestamp": timestamp
    }
    
    # Save detailed results
    with open(f"rouge_evaluation_results_{timestamp}.json", "w", encoding='utf-8') as f:
        json.dump({"summary": summary, "detailed_results": results}, f, indent=2)
    
    # Save summary as CSV for easy analysis
    df = pd.DataFrame([{
        "article": r["article"],
        "reference_summary": r["reference_summary"],
        "model_summary": r["model_summary"],
        **r["rouge_scores"]
    } for r in results])
    df.to_csv(f"rouge_evaluation_results_{timestamp}.csv", index=False, encoding='utf-8')
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total examples evaluated: {len(results)}")
    print("\nAverage ROUGE Scores:")
    for metric, score in avg_scores.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()