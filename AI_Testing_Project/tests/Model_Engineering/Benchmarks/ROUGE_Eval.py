import anthropic
from rouge_score import rouge_scorer
import json
import time
from typing import List, Dict, Any
from tqdm import tqdm
import os

class RougeEvaluator:
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.2,
        max_tokens: int = 1000
    ):
        self.client = anthropic.Client(api_key=os.getenv('$ANTHROPIC_API_KEY'))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Initialize ROUGE scorer with all ROUGE variants
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
    def load_test_data(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load test data from a JSON file containing prompts and reference responses.
        Expected format:
        [
            {
                "prompt": "Summarize this text: ...",
                "reference": "Reference summary..."
            },
            ...
        ]
        """
        file_path="C:\\Users\\SandraDujmovic\\Documents\\AI_Testing_Projects\\ClaudeAI\\tests\\LLM\\Model_Engineering\\Benchmarks\\Dataset\\rouge_test_data.json"

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_model_response(self, prompt: str) -> str:
        """
        Get response from the model for a given prompt.
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Error getting model response: {str(e)}")
            return ""

    def calculate_rouge_scores(self, prediction: str, reference: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE scores between prediction and reference.
        """
        try:
            scores = self.scorer.score(reference, prediction)
            # Convert Score objects to regular dictionaries
            return {
                metric: {
                    'precision': scores[metric].precision,
                    'recall': scores[metric].recall,
                    'fmeasure': scores[metric].fmeasure
                }
                for metric in scores
            }
        except Exception as e:
            print(f"Error calculating ROUGE scores: {str(e)}")
            return {}

    def evaluate_batch(self, test_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate model on a batch of test cases and calculate ROUGE scores.
        """
        results = {
            "individual_scores": [],
            "average_scores": {
                "rouge1": {"precision": 0, "recall": 0, "fmeasure": 0},
                "rouge2": {"precision": 0, "recall": 0, "fmeasure": 0},
                "rougeL": {"precision": 0, "recall": 0, "fmeasure": 0}
            }
        }
        
        for test_case in tqdm(test_data, desc="Evaluating"):
            try:
                # Get model prediction
                prediction = self.get_model_response(test_case["prompt"])
                
                # Calculate ROUGE scores
                scores = self.calculate_rouge_scores(prediction, test_case["reference"])
                
                # Store individual results
                results["individual_scores"].append({
                    "prompt": test_case["prompt"],
                    "reference": test_case["reference"],
                    "prediction": prediction,
                    "scores": scores
                })
                
                # Update running averages
                for metric in scores:
                    for score_type in scores[metric]:
                        results["average_scores"][metric][score_type] += \
                            scores[metric][score_type] / len(test_data)
                
                # Add a small delay to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing test case: {str(e)}")
                continue
        
        return results

    def save_results(self, results: Dict[str, Any], output_file: str = "rouge_results.json"):
        """
        Save evaluation results to a JSON file.
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

def print_summary(results: Dict[str, Any]):
    """
    Print a summary of the ROUGE evaluation results.
    """
    print("\nROUGE Score Summary:")
    print("-" * 50)
    for metric, scores in results["average_scores"].items():
        print(f"\n{metric.upper()}:")
        for score_type, value in scores.items():
            print(f"  {score_type}: {value:.4f}")

def main():
    # Initialize evaluator
    evaluator = RougeEvaluator()
    
    # Load test data
    test_data = evaluator.load_test_data("rouge_test_data.json")
    
    # Run evaluation
    results = evaluator.evaluate_batch(test_data)
    
    # Save results
    evaluator.save_results(results)
    
    # Print summary
    print_summary(results)

if __name__ == "__main__":
    main()