import anthropic
import json
import time
from typing import List, Dict, Any
from tqdm import tqdm
import os
from pathlib import Path
import sys

# Add config directory to path
config_path = Path(__file__).parent.parent.parent / 'config'
sys.path.append(str(config_path))

from config import * 

def load_human_eval_problems(file_path: str = "HumanEval.jsonl") -> List[Dict[str, Any]]:
    """
    Load HumanEval problems from a JSON file.
    """
    problems = []
     with open(f"{Config.dataset_folder}/humaneval_dataset.json", "r") as f:   
        for line in f:
            problems.append(json.loads(line))
    return problems

def create_prompt(problem: Dict[str, Any]) -> str:
    """
    Create a prompt for the model using the problem description and function signature.
    """
    return f"""Complete the following Python function:

{problem['prompt']}

The function should:
{problem['description']}

Please provide only the function implementation without any explanation or tests."""

def evaluate_solution(problem: Dict[str, Any], completion: str) -> bool:
    """
    Evaluate if the model's solution passes the test cases.
    Note: This is a simplified evaluation. In practice, you'd want to run the tests
    in a sandboxed environment.
    """
    try:
        # Combine the original prompt with the completion
        full_code = problem['prompt'] + completion
        
        # Create a new namespace to avoid polluting the global namespace
        namespace = {}
        
        # Execute the function definition
        exec(full_code, namespace)
        
        # Execute the test cases
        test_code = problem['test']
        exec(test_code, namespace)
        
        return True
    except Exception as e:
        print(f"Error evaluating solution: {str(e)}")
        return False

def run_human_eval_benchmark(
    model: str = "claude-3-5-sonnet-20241022",
    num_problems: int = 10,
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Run the HumanEval benchmark on the specified model.
    """
    client = anthropic.Client()
    problems = load_human_eval_problems()[:num_problems]
    
    results = {
        "total_problems": num_problems,
        "successful": 0,
        "failed": 0,
        "errors": 0,
        "details": []
    }
    
    for problem in tqdm(problems, desc="Evaluating problems"):
        try:
            prompt = create_prompt(problem)
            
            # Get completion from Claude
            response = client.messages.create(
                model=model,
                max_tokens=1500,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            completion = response.content[0].text
            
            # Evaluate the solution
            success = evaluate_solution(problem, completion)
            
            results["details"].append({
                "task_id": problem["task_id"],
                "success": success,
                "completion": completion
            })
            
            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1
                
            # Add a small delay to avoid rate limits
            time.sleep(1)
            
        except Exception as e:
            results["errors"] += 1
            results["details"].append({
                "task_id": problem["task_id"],
                "error": str(e)
            })
    
    # Calculate success rate
    results["success_rate"] = results["successful"] / results["total_problems"]
    
    return results

def save_results(results: Dict[str, Any], output_file: str = "human_eval_results.json"):
    """
    Save benchmark results to a JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    # Run the benchmark
    results = run_human_eval_benchmark()
    
    # Save results
    save_results(results)
    
    # Print summary
    print(f"\nBenchmark Results:")
    print(f"Total Problems: {results['total_problems']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Errors: {results['errors']}")
    print(f"Success Rate: {results['success_rate']*100:.2f}%")
