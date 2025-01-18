import anthropic
from bert_score import BERTScorer
import json
import time
from typing import List, Dict, Any
from tqdm import tqdm
import torch
import numpy as np
import os

class BertScoreEvaluator:
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.2,
        max_tokens: int = 1000,
        bert_model: str = "roberta-large",
        device: str = None,
        lang: str = "en"
    ):
        self.client = anthropic.Client(api_key=os.getenv('$ANTHROPIC_API_KEY'))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set device (CPU/GPU)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize BERTScore
        print(f"Initializing BERTScorer with {bert_model} on {self.device}...")
        self.scorer = BERTScorer(
            model_type=bert_model,
            num_layers=17,  # Using the second-to-last layer
            batch_size=32,
            device=self.device,
            lang=lang,
            rescale_with_baseline=True
        )
        
    def load_test_data(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load test data from a JSON file containing prompts and reference responses.
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

    def calculate_bert_scores(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, List[float]]:
        """
        Calculate BERTScore metrics for a batch of predictions and references.
        """
        try:
            P, R, F1 = self.scorer.score(predictions, references)
            return {
                "precision": P.tolist(),
                "recall": R.tolist(),
                "f1": F1.tolist()
            }
        except Exception as e:
            print(f"Error calculating BERT scores: {str(e)}")
            return {
                "precision": [],
                "recall": [],
                "f1": []
            }

    def evaluate_batch(self, test_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate model on a batch of test cases and calculate BERT scores.
        """
        results = {
            "individual_scores": [],
            "average_scores": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            },
            "metadata": {
                "model": self.model,
                "temperature": self.temperature,
                "bert_model": self.scorer.model_type,
                "device": self.device,
                "num_samples": len(test_data)
            }
        }
        
        # Collect predictions and references
        predictions = []
        references = []
        prompts = []
        
        print("Generating model responses...")
        for test_case in tqdm(test_data):
            prediction = self.get_model_response(test_case["prompt"])
            predictions.append(prediction)
            references.append(test_case["reference"])
            prompts.append(test_case["prompt"])
            time.sleep(1)  # Rate limiting
            
        print("Calculating BERT scores...")
        batch_scores = self.calculate_bert_scores(predictions, references)
        
        # Store individual results
        for i in range(len(predictions)):
            results["individual_scores"].append({
                "prompt": prompts[i],
                "reference": references[i],
                "prediction": predictions[i],
                "scores": {
                    "precision": batch_scores["precision"][i],
                    "recall": batch_scores["recall"][i],
                    "f1": batch_scores["f1"][i]
                }
            })
        
        # Calculate averages
        results["average_scores"]["precision"] = np.mean(batch_scores["precision"])
        results["average_scores"]["recall"] = np.mean(batch_scores["recall"])
        results["average_scores"]["f1"] = np.mean(batch_scores["f1"])
        
        # Add score distributions
        results["score_distributions"] = {
            "precision": {
                "min": float(np.min(batch_scores["precision"])),
                "max": float(np.max(batch_scores["precision"])),
                "std": float(np.std(batch_scores["precision"]))
            },
            "recall": {
                "min": float(np.min(batch_scores["recall"])),
                "max": float(np.max(batch_scores["recall"])),
                "std": float(np.std(batch_scores["recall"]))
            },
            "f1": {
                "min": float(np.min(batch_scores["f1"])),
                "max": float(np.max(batch_scores["f1"])),
                "std": float(np.std(batch_scores["f1"]))
            }
        }
        
        return results

    def save_results(self, results: Dict[str, Any], output_file: str = "bert_score_results.json"):
        """
        Save evaluation results to a JSON file.
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

def print_summary(results: Dict[str, Any]):
    """
    Print a summary of the BERTScore evaluation results.
    """
    print("\nBERTScore Evaluation Summary:")
    print("-" * 50)
    print(f"\nModel: {results['metadata']['model']}")
    print(f"BERT model: {results['metadata']['bert_model']}")
    print(f"Number of samples: {results['metadata']['num_samples']}")
    print(f"Device: {results['metadata']['device']}")
    
    print("\nAverage Scores:")
    for metric, value in results["average_scores"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nScore Distributions:")
    for metric, stats in results["score_distributions"].items():
        print(f"\n{metric}:")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
        print(f"  Std: {stats['std']:.4f}")

def main():
    # Initialize evaluator
    evaluator = BertScoreEvaluator()
    
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


