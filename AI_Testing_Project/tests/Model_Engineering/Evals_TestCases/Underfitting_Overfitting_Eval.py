from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import numpy as np
import json
import time
from typing import Dict

# Set up logging and style configurations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot style
sns.set_theme()
sns.set_palette("husl")

class ClaudeFitTester:
    def __init__(self, api_key: str):
        """Initialize the ClaudeFitTester with API key and default parameters.
        
        Args:
            api_key (str): The Anthropic API key for accessing Claude
        """
        self.api_key = api_key
        
    def run_evaluation(self, num_iterations: int = 3):
        """Run the evaluation process.
        
        Args:
            num_iterations (int): Number of evaluation iterations
            
        Returns:
            Dict: Results of the evaluation
        """
        # Placeholder for evaluation logic
        results = {
            "overall_metrics": {
                "mean_scores": {"simple": 0.8, "medium": 0.7, "complex": 0.6},
                "std_scores": {"simple": 0.1, "medium": 0.15, "complex": 0.2}
            },
            "type_performance": {
                "basic": {"mean": 0.85},
                "intermediate": {"mean": 0.75},
                "advanced": {"mean": 0.65}
            },
            "fit_analysis": {
                "underfitting_indicators": {
                    "simple_tasks_performance": 0.8,
                    "variance_in_basic_tasks": 0.1
                },
                "overfitting_indicators": {
                    "complex_vs_novel_ratio": 0.7,
                    "edge_case_performance": 0.6
                }
            },
            "conclusions": {
                "underfitting_risk": "low",
                "overfitting_risk": "medium"
            }
        }
        return results

    def generate_visualizations(self, results: Dict, output_dir: str = "results"):
        """Generate and save visualization plots."""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # 1. Complexity Performance Plot
        self._plot_complexity_performance(results, output_dir, timestamp)
        
        # 2. Task Type Performance Plot
        self._plot_task_type_performance(results, output_dir, timestamp)
        
        # 3. Performance Distribution Plot
        self._plot_performance_distribution(results, output_dir, timestamp)
        
        # 4. Fitting Analysis Plot
        self._plot_fitting_analysis(results, output_dir, timestamp)

    def _plot_complexity_performance(self, results: Dict, output_dir: str, timestamp: str):
        """Create bar plot of performance across complexity levels."""
        plt.figure(figsize=(12, 6))
        
        complexities = list(results["overall_metrics"]["mean_scores"].keys())
        means = [results["overall_metrics"]["mean_scores"][c] for c in complexities]
        stds = [results["overall_metrics"]["std_scores"][c] for c in complexities]
        
        bars = plt.bar(complexities, means, yerr=stds, capsize=5)
        plt.title("Performance Across Complexity Levels", pad=20)
        plt.ylabel("Mean Score")
        plt.ylim(0, 1.2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/complexity_performance_{timestamp}.png")
        plt.close()

    def _plot_task_type_performance(self, results: Dict, output_dir: str, timestamp: str):
        """Create radar plot of performance across task types."""
        task_types = list(results["type_performance"].keys())
        means = [results["type_performance"][t]["mean"] for t in task_types]
        
        # Create radar plot
        angles = np.linspace(0, 2*np.pi, len(task_types), endpoint=False)
        means = np.concatenate((means, [means[0]]))  # complete the circle
        angles = np.concatenate((angles, [angles[0]]))  # complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, means)
        ax.fill(angles, means, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(task_types)
        ax.set_ylim(0, 1)
        plt.title("Performance by Task Type", pad=20)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/task_type_performance_{timestamp}.png")
        plt.close()

    def _plot_performance_distribution(self, results: Dict, output_dir: str, timestamp: str):
        """Create violin plot showing score distributions."""
        plt.figure(figsize=(12, 6))
        
        data_dict = {
            complexity: results["overall_metrics"]["mean_scores"][complexity]
            for complexity in results["overall_metrics"]["mean_scores"]
        }
        
        plt.violinplot(list(data_dict.values()))
        
        plt.title("Score Distribution Across Complexity Levels", pad=20)
        plt.xticks(range(1, len(data_dict) + 1), data_dict.keys(), rotation=45)
        plt.ylabel("Score Distribution")
        plt.ylim(0, 1.2)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_distribution_{timestamp}.png")
        plt.close()

    def _plot_fitting_analysis(self, results: Dict, output_dir: str, timestamp: str):
        """Create visualization of fitting analysis."""
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        metrics = {
            'Simple Tasks\nPerformance': results["fit_analysis"]["underfitting_indicators"]["simple_tasks_performance"],
            'Variance in\nBasic Tasks': results["fit_analysis"]["underfitting_indicators"]["variance_in_basic_tasks"],
            'Complex vs\nNovel Ratio': results["fit_analysis"]["overfitting_indicators"]["complex_vs_novel_ratio"],
            'Edge Case\nPerformance': results["fit_analysis"]["overfitting_indicators"]["edge_case_performance"]
        }
        
        # Create horizontal bar plot
        bars = plt.barh(list(metrics.keys()), list(metrics.values()))
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, 
                    bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}',
                    ha='left', 
                    va='center')
        
        plt.title("Fitting Analysis Metrics", pad=20)
        plt.xlabel("Score")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fitting_analysis_{timestamp}.png")
        plt.close()

def main():
    # Replace with your API key
    api_key = os.getenv('$ANTHROPIC_API_KEY')
    if api_key is None:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")
    
    # Initialize tester
    tester = ClaudeFitTester(api_key)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = tester.run_evaluation(num_iterations=3)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    tester.generate_visualizations(results)
    
    # Save results
    output_file = f"results/claude_fit_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    Path("results").mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print key findings
    logger.info("\nKey Findings:")
    logger.info(f"Underfitting Risk: {results['conclusions']['underfitting_risk']}")
    logger.info(f"Overfitting Risk: {results['conclusions']['overfitting_risk']}")
    logger.info("\nPerformance by Complexity:")
    for complexity, score in results["overall_metrics"]["mean_scores"].items():
        logger.info(f"{complexity}: {score:.3f}")

if __name__ == "__main__":
    main()