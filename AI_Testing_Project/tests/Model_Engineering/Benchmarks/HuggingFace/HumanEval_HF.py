from datasets import load_dataset
import anthropic
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
import ast
import contextlib
import io
import sys
import multiprocessing
from multiprocessing import Pool
import signal
import os

def load_humaneval():
    """
    Load HumanEval dataset from HuggingFace
    """
    dataset = load_dataset("openai_humaneval")
    return dataset["test"]

def format_prompt(task):
    """
    Format the coding task for Claude
    """
    prompt = f"""Complete the following Python function according to the given docstring and test cases.
    Only provide the function implementation, without any test cases or additional explanation.
    Do not include the docstring in your response.

    {task['prompt']}
    """
    return prompt

def get_model_response(client, prompt):
    """
    Get code completion from Claude 3.5 Sonnet
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

def extract_code(response):
    """
    Extract code from model's response, handling potential markdown formatting
    """
    # Remove markdown code blocks if present
    code = response.replace("```python", "").replace("```", "").strip()
    return code

def run_test_with_timeout(args):
    """
    Run a single test case with timeout using multiprocessing
    """
    function_code, test_case, entry_point = args
    
    def run_test():
        namespace = {}
        try:
            # Execute the function definition
            exec(function_code, namespace)
            # Execute the test case
            exec(f"result = {test_case}", namespace)
            return namespace["result"]
        except Exception as e:
            return False

    # Use multiprocessing Pool with timeout
    with Pool(1) as pool:
        try:
            result = pool.apply_async(run_test)
            return result.get(timeout=5)  # 5 second timeout
        except Exception as e:
            return False
        finally:
            pool.terminate()
            pool.join()

def evaluate_solution(function_code, entry_point, test_cases):
    """
    Evaluate the model's solution against test cases
    """
    results = []
    
    for test_case in test_cases:
        try:
            result = run_test_with_timeout((function_code, test_case, entry_point))
            results.append(result)
        except Exception as e:
            results.append(False)
    
    return all(results), results

def serialize_result(result):
    """
    Convert result dictionary to JSON-serializable format
    """
    return {
        "task_id": str(result["task_id"]),
        "prompt": str(result["prompt"]),
        "model_code": str(result["model_code"]),
        "passed_tests": result["passed_tests"],
        "test_results": result["test_results"],
        "error": str(result.get("error", ""))
    }

def main():
    # Initialize multiprocessing support for Windows
    multiprocessing.freeze_support()
    
    # Initialize Anthropic client
    client = anthropic.Client(api_key=os.getenv('$ANTHROPIC_API_KEY'))
    
    # Load dataset
    dataset = load_humaneval()
    
    # Initialize results storage
    results = []
    
    # Create timestamp for the evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Evaluate each task
    for idx in tqdm(range(len(dataset))):
        try:
            task = dataset[idx]
            
            # Format prompt and get model's response
            prompt = format_prompt(task)
            model_response = get_model_response(client, prompt)
            
            # Extract code from response
            model_code = extract_code(model_response)
            
            # Evaluate solution
            passed_all, test_results = evaluate_solution(
                model_code,
                task["entry_point"],
                task["test"]
            )
            
            # Store results
            result = {
                "task_id": task["task_id"],
                "prompt": task["prompt"],
                "model_code": model_code,
                "passed_tests": passed_all,
                "test_results": test_results
            }
            results.append(serialize_result(result))
            
        except Exception as e:
            print(f"Error processing task {idx}: {str(e)}")
            result = {
                "task_id": task["task_id"],
                "prompt": task["prompt"],
                "model_code": "",
                "passed_tests": False,
                "test_results": [],
                "error": str(e)
            }
            results.append(serialize_result(result))
            continue
    
    # Calculate metrics
    total_tasks = len(results)
    passed_tasks = sum(1 for r in results if r["passed_tests"])
    pass_rate = passed_tasks / total_tasks if total_tasks > 0 else 0
    
    # Create summary
    summary = {
        "model": "claude-3-5-sonnet-20241022",
        "total_tasks": total_tasks,
        "passed_tasks": passed_tasks,
        "pass_rate": pass_rate,
        "timestamp": timestamp
    }
    
    # Save detailed results
    with open(f"humaneval_results_{timestamp}.json", "w", encoding='utf-8') as f:
        json.dump({"summary": summary, "detailed_results": results}, f, indent=2)
    
    # Save summary as CSV for easy analysis
    df = pd.DataFrame(results)
    df.to_csv(f"humaneval_results_{timestamp}.csv", index=False, encoding='utf-8')
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total tasks evaluated: {total_tasks}")
    print(f"Tasks passed: {passed_tasks}")
    print(f"Pass rate: {pass_rate:.2%}")

if __name__ == "__main__":
    main()