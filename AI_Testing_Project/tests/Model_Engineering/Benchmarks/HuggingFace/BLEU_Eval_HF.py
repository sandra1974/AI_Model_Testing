import os
from datasets import load_dataset
from anthropic import Anthropic
import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

def download_nltk_data():
    """Download all required NLTK data"""
    try:
        # Download punkt tokenizer data
        nltk.download('punkt')
        # Additional downloads if needed
        nltk.download('averaged_perceptron_tagger')
        nltk.download('universal_tagset')
        
        # Verify the download
        nltk.data.find('tokenizers/punkt')
        print("NLTK resources downloaded successfully")
        
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        raise

def simple_tokenize(text):
    """Simple whitespace tokenization as fallback"""
    return text.split()

def evaluate_bleu(api_key, dataset_name, dataset_config=None, num_samples=None):
    """
    Evaluate Claude's BLEU score on a dataset
    
    Args:
        api_key (str): Anthropic API key
        dataset_name (str): HuggingFace dataset name
        dataset_config (str, optional): Dataset configuration name
        num_samples (int, optional): Number of samples to evaluate
    """
    # Download NLTK data first
    try:
        download_nltk_data()
    except Exception as e:
        print(f"Warning: NLTK download failed, using simple tokenization: {e}")
    
    # Initialize Anthropic client
    client = Anthropic(api_key=os.getenv('$ANTHROPIC_API_KEY'))
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_name, dataset_config)["test"]
        print(f"\nLoaded dataset with {len(dataset)} examples")
        
        # Print first example
        print("\nFirst example structure:")
        print(dataset[0])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    results = []
    smoothing = SmoothingFunction().method1
    
    # Extract language pair from config
    source_lang = dataset_config.split('-')[1]  # 'de-en' -> 'en'
    target_lang = dataset_config.split('-')[0]  # 'de-en' -> 'de'
    
    print(f"\nTranslating from {source_lang} to {target_lang}")
    
    for item in tqdm(dataset):
        try:
            # Extract source and target texts
            source_text = item['translation'][source_lang]
            reference_text = item['translation'][target_lang]
            
            # Create prompt
            prompt = f"Translate this {source_lang} text to {target_lang}: {source_text}"
            
            # Get model's response
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            model_output = response.content[0].text.strip()
            
            # Try NLTK tokenization first, fall back to simple tokenization if it fails
            try:
                reference_tokens = [nltk.word_tokenize(reference_text.lower())]
                hypothesis_tokens = nltk.word_tokenize(model_output.lower())
            except Exception as e:
                print(f"Warning: NLTK tokenization failed, using simple tokenization: {e}")
                reference_tokens = [simple_tokenize(reference_text.lower())]
                hypothesis_tokens = simple_tokenize(model_output.lower())
            
            bleu_score = sentence_bleu(
                reference_tokens,
                hypothesis_tokens,
                smoothing_function=smoothing
            )
            
            # Store results
            results.append({
                'source_text': source_text,
                'reference_text': reference_text,
                'model_output': model_output,
                'bleu_score': bleu_score
            })
            
            # Print progress for first example
            if len(results) == 1:
                print("\nFirst translation example:")
                print(f"Source ({source_lang}): {source_text}")
                print(f"Reference ({target_lang}): {reference_text}")
                print(f"Model output: {model_output}")
                print(f"BLEU score: {bleu_score}")
            
        except Exception as e:
            print(f"\nError processing item: {str(e)}")
            print(f"Source text: {source_text}")
            print(f"Reference text: {reference_text}")
            continue
    
    if not results:
        print("No results generated. Please check the dataset structure.")
        return None
        
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate average BLEU score
    avg_bleu = results_df['bleu_score'].mean()
    
    return {
        'average_bleu': avg_bleu,
        'detailed_results': results_df
    }

if __name__ == "__main__":
    # Get API key from environment
    API_KEY = os.getenv('$ANTHROPIC_API_KEY')
    
    if not API_KEY:
        print("Please set your ANTHROPIC_API_KEY environment variable")
        exit(1)
    
    # Run evaluation
    try:
        results = evaluate_bleu(
            api_key=API_KEY,
            dataset_name="wmt16",
            dataset_config="de-en",
            num_samples=15
        )
        
        if results:
            print(f"\nEvaluation Results:")
            print(f"Average BLEU Score: {results['average_bleu']:.4f}")
            
            # Save results
            output_file = 'bleu_evaluation_results.csv'
            results['detailed_results'].to_csv(output_file, index=False)
            print(f"\nDetailed results saved to {output_file}")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")