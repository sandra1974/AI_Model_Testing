import anthropic
from datasets import load_dataset
from bert_score import score
from tqdm import tqdm
import numpy as np
import json
from typing import List, Dict, Tuple
import time
import logging
import os


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClaudeEvaluator:
    def __init__(self, api_key: str):
        """Initialize the evaluator with Anthropic API key."""
        self.client = anthropic.Client(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def get_claude_response(self, prompt: str) -> str:
        """Get response from Claude with error handling and retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                # Convert content to string, handling both TextBlock and string cases
                response_content = str(message.content[0].text) if hasattr(message.content[0], 'text') else str(message.content)
                return response_content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get Claude response after {max_retries} attempts: {e}")
                    return ""
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff

    def clean_text(self, text: any) -> str:
        """Clean and convert any text input to string."""
        if hasattr(text, 'text'):  # Handle TextBlock objects
            return str(text.text).strip()
        elif isinstance(text, list):  # Handle list inputs
            return ' '.join(str(t) for t in text).strip()
        else:  # Handle string or other inputs
            return str(text).strip()

    def calculate_bert_scores(
        self, 
        generated_texts: List[str], 
        reference_texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate BERTScore metrics."""
        # Clean and ensure all inputs are strings
        generated_texts = [self.clean_text(text) for text in generated_texts]
        reference_texts = [self.clean_text(text) for text in reference_texts]
        
        # Filter out empty responses
        valid_pairs = [(gen, ref) for gen, ref in zip(generated_texts, reference_texts) if gen and ref]
        if not valid_pairs:
            raise ValueError("No valid text pairs found for scoring")
        
        generated_texts, reference_texts = zip(*valid_pairs)
        
        # Calculate BERTScore
        P, R, F1 = score(
            generated_texts,
            reference_texts,
            lang="en",
            verbose=True
        )
        
        return P, R, F1

    def evaluate_dataset(
        self, 
        dataset_name: str, 
        split: str = None,
        prompt_column: str = "prompt",
        reference_column: str = "response",
        num_samples: int = None
    ) -> Dict:
        """
        Evaluate Claude on a HuggingFace dataset.
        """
        # Load dataset with split validation
        logger.info(f"Loading dataset: {dataset_name}")
        
        dataset = None
        if split is None:
            for try_split in ['test', 'validation', 'train']:
                try:
                    dataset = load_dataset(dataset_name, split=try_split)
                    split = try_split
                    logger.info(f"Successfully loaded {try_split} split")
                    break
                except ValueError:
                    continue
        else:
            try:
                dataset = load_dataset(dataset_name, split=split)
            except ValueError as e:
                logger.error(f"Error loading specified split: {e}")
                available_splits = load_dataset(dataset_name).keys()
                raise ValueError(f"Specified split '{split}' not found. Available splits: {list(available_splits)}")
        
        if dataset is None:
            raise ValueError(f"Could not load any split from dataset {dataset_name}")
        
        # Verify column names
        if prompt_column not in dataset.column_names:
            raise ValueError(f"Prompt column '{prompt_column}' not found. Available columns: {dataset.column_names}")
        if reference_column not in dataset.column_names:
            raise ValueError(f"Reference column '{reference_column}' not found. Available columns: {dataset.column_names}")
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        generated_responses = []
        reference_responses = []
        
        # Generate responses
        logger.info("Generating responses from Claude")
        for item in tqdm(dataset):
            prompt = str(item[prompt_column])
            reference = str(item[reference_column])
            
            response = self.get_claude_response(prompt)
            generated_responses.append(response)
            reference_responses.append(reference)
        
        # Calculate scores
        logger.info("Calculating BERTScores")
        P, R, F1 = self.calculate_bert_scores(generated_responses, reference_responses)
        
        # Prepare results
        results = {
            "dataset": dataset_name,
            "split": split,
            "model": self.model,
            "num_samples": len(dataset),
            "metrics": {
                "bert_score_precision": float(P.mean()),
                "bert_score_recall": float(R.mean()),
                "bert_score_f1": float(F1.mean())
            },
            "samples": [
                {
                    "prompt": str(item[prompt_column]),
                    "reference": str(item[reference_column]),
                    "generated": gen,
                    "scores": {
                        "precision": float(p),
                        "recall": float(r),
                        "f1": float(f1)
                    }
                }
                for item, gen, p, r, f1 in zip(
                    dataset, 
                    generated_responses, 
                    P, R, F1
                )
            ]
        }
        
        return results

def main():
    # Replace with your Anthropic API key
    api_key = os.getenv('$ANTHROPIC_API_KEY')
    
    # Initialize evaluator
    evaluator = ClaudeEvaluator(api_key)
    
    # Example usage with a HuggingFace dataset
    dataset_name = "databricks/databricks-dolly-15k"
    results = evaluator.evaluate_dataset(
        dataset_name=dataset_name,
        split=None,  # Will automatically try test, validation, then train
        prompt_column="instruction",
        reference_column="response",
        num_samples=5
    )
    
    # Save results
    output_file = f"bert_evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {output_file}")
    logger.info(f"Average BERTScore F1: {results['metrics']['bert_score_f1']:.4f}")

if __name__ == "__main__":
    main()
