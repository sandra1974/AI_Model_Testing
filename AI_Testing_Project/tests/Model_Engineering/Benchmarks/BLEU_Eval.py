import anthropic
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
from typing import List, Dict
import os
from tqdm import tqdm
import logging
import os
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationEvaluator:
    def __init__(self, api_key: str):
        """Initialize the evaluator with API credentials."""
        self.client = anthropic.Client(api_key=os.getenv('$ANTHROPIC_API_KEY'))
        self.model = "claude-3-5-sonnet-20241022"
        self.smoothing = SmoothingFunction().method1

    def get_translation(self, text: str, source_lang: str, target_lang: str) -> str:
        """Get translation from Claude."""
        try:
            prompt = f"""Translate the following text from {source_lang} to {target_lang}. 
            Provide only the translation with no additional explanation:
            
            {text}"""
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise

    def calculate_bleu(self, candidate: str, references: List[str]) -> float:
        """Calculate BLEU score for a single translation."""
        candidate_tokens = candidate.lower().split()
        reference_tokens = [ref.lower().split() for ref in references]
        
        return sentence_bleu(
            reference_tokens,
            candidate_tokens,
            smoothing_function=self.smoothing
        )

    def evaluate_translations(self, test_data: List[Dict]) -> Dict:
        """Evaluate translations for a test dataset."""
        results = []
        total_bleu = 0.0
        
        for item in tqdm(test_data, desc="Evaluating translations"):
            try:
                source_text = item['source']
                reference_translations = item['references']
                source_lang = item['source_lang']
                target_lang = item['target_lang']
                
                # Get Claude's translation
                claude_translation = self.get_translation(
                    source_text,
                    source_lang,
                    target_lang
                )
                
                # Calculate BLEU score
                bleu_score = self.calculate_bleu(
                    claude_translation,
                    reference_translations
                )
                
                results.append({
                    'source': source_text,
                    'claude_translation': claude_translation,
                    'references': reference_translations,
                    'bleu_score': bleu_score
                })
                
                total_bleu += bleu_score
                
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
                continue
        
        avg_bleu = total_bleu / len(results) if results else 0
        
        return {
            'detailed_results': results,
            'average_bleu': avg_bleu
        }

def load_test_data(file_path: str) -> List[Dict]:
    """Load test cases from a JSON file."""
    file_path="C:\\Users\\SandraDujmovic\\Documents\\AI_Testing_Projects\\ClaudeAI\\tests\\LLM\\Model_Engineering\\Benchmarks\\Dataset\\translation_test_data.json"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Validate the data structure
        required_fields = {'source', 'references', 'source_lang', 'target_lang'}
        for i, item in enumerate(test_data):
            missing_fields = required_fields - set(item.keys())
            if missing_fields:
                raise ValueError(f"Test case {i} is missing required fields: {missing_fields}")
        
        return test_data
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate Claude translations using BLEU score')
    parser.add_argument('--input', '-i', required=True, help='Path to input JSON file containing test cases')
    parser.add_argument('--output', '-o', default='translation_results.json', help='Path to output JSON file')
    args = parser.parse_args()
    
    # Initialize evaluator with your API key
    api_key = os.getenv('$ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
    
    try:
        # Load test cases from JSON file
        logger.info(f"Loading test cases from {args.input}")
        test_data = load_test_data(args.input)
        logger.info(f"Loaded {len(test_data)} test cases")
        
        # Initialize evaluator and run evaluation
        evaluator = TranslationEvaluator(api_key)
        results = evaluator.evaluate_translations(test_data)
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Average BLEU score: {results['average_bleu']:.4f}")
        logger.info(f"Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

