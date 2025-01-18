from datasets import load_dataset
import anthropic
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
import os

def load_hellaswag():
    """
    Load HellaSwag dataset from HuggingFace with trust_remote_code=True
    """
    dataset = load_dataset("hellaswag", trust_remote_code=True)
    return dataset["validation"]  # Using validation set for evaluation

def format_prompt(context, endings):
    """
    Format the prompt for Claude with context and possible endings
    """
    prompt = f"Given the context: '{context}'\n\n"
    prompt += "Which of the following is the most likely continuation?\n\n"
    
    for i, ending in enumerate(endings):
        prompt += f"{chr(65 + i)}. {ending}\n"
    
    return prompt

def get_model_response(client, prompt):
    """
    Get response from Claude 3.5 Sonnet
    """
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        temperature=0,  # Using temperature 0 for deterministic responses
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return str(message.content)

def parse_model_answer(response):
    """
    Parse model's response to extract the selected option (A, B, C, or D)
    """
    response = response.upper()
    for option in ['A', 'B', 'C', 'D']:
        if option in response[:10]:  # Check first 10 characters for the answer
            return ord(option) - ord('A')  # Convert letter to index (0-3)
    return None

def serialize_result(result):
    """
    Convert result dictionary to JSON-serializable format
    """
    return {
        "context": str(result["context"]),
        "correct_ending": str(result["correct_ending"]),
        "endings": [str(ending) for ending in result["endings"]],
        "model_response": str(result["model_response"]),
        "model_choice": result["model_choice"],
        "correct_choice": result["correct_choice"],
        "is_correct": result["is_correct"],
        "score": result["score"]
    }

def main():
    # Initialize Anthropic client
    client = anthropic.Client(api_key=os.getenv('$ANTHROPIC_API_KEY'))
    
    # Load dataset
    dataset = load_hellaswag()
    
    # Initialize results storage
    results = []
    
    # Create timestamp for the evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Evaluate each question
    for idx in tqdm(range(len(dataset))):
        try:
            # Get the context and endings
            context = dataset[idx]["ctx"]
            endings = dataset[idx]["endings"]
            correct_idx = int(dataset[idx]["label"])
            
            # Format prompt
            prompt = format_prompt(context, endings)
            
            # Get model response
            model_response = get_model_response(client, prompt)
            
            # Parse model's answer
            model_choice = parse_model_answer(model_response)
            
            # Calculate score
            is_correct = model_choice == correct_idx
            score = 1 if is_correct else 0
            
            # Store results
            result = {
                "context": context,
                "correct_ending": endings[correct_idx],
                "endings": endings,
                "model_response": model_response,
                "model_choice": model_choice,
                "correct_choice": correct_idx,
                "is_correct": is_correct,
                "score": score
            }
            results.append(serialize_result(result))
            
        except Exception as e:
            print(f"Error processing example {idx}: {str(e)}")
            continue
    
    # Calculate overall metrics
    total_examples = len(results)
    correct_answers = sum(r["score"] for r in results)
    accuracy = correct_answers / total_examples if total_examples > 0 else 0
    
    # Create summary
    summary = {
        "model": "claude-3-5-sonnet-20241022",
        "total_examples": total_examples,
        "correct_answers": correct_answers,
        "accuracy": accuracy,
        "timestamp": timestamp
    }
    
    # Save detailed results
    with open(f"hellaswag_results_{timestamp}.json", "w", encoding='utf-8') as f:
        json.dump({"summary": summary, "detailed_results": results}, f, indent=2)
    
    # Save summary as CSV for easy analysis
    df = pd.DataFrame(results)
    df.to_csv(f"hellaswag_results_{timestamp}.csv", index=False, encoding='utf-8')
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total examples evaluated: {total_examples}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()

