# AI_Model_Testing
This project aims to validate how an AI model behaves in different scenarios and whether it returns exact, accurate, and relevant answers. It is also tested how the AI model behaves when many users simultaneously ask questions and what is the speed of the AI model response. All these tests are conducted on AI model claude-3-5-sonnet-20241022. Test scripts are divided into two separate folders: 

•	`Model_Engineering`<br/> 
•	`Model_Evaluation`

During the model engineering phase, benchmark tests and overfitting and underfitting evaluations are completed, while functional, performance, and API tests are executed in the model evaluation phase. However, API tests can also be performed at an earlier stage.

## Prerequisites
The following environment is set up:<br/>

•	Visual Studio<br/>
•	Python<br/>
•	Postman<br/>
•	JMeter<br/>
•	API key generated from this link: https://console.anthropic.com/dashboard<br/> 
•	Configured environment variable > > > variable name: $ANTHROPIC_API_KEY and variable value: api key<br/>

## Benchmark Testing<br/>
Benchmark tests are written in Python and completed in 2 ways:<br/>

•	Using dataset files stored in `Dataset` folder<br/>
•	Using the HuggingFace dataset<br/> 

All Benchmark tests are stored inside `Model_Engineering` folder.<br/>

### MMLU Evaluation<br/>
MMLU (Massive Multitask Language Understanding) is a testing tool that evaluates how well AI language models perform across 57 different subjects. It works like a comprehensive exam, using multiple-choice questions to test knowledge in areas from basic science to advanced professional topics like law and medicine. Its main purpose is to measure how well AI models understand and can reason about different subjects, and to provide a way to compare different AI models' capabilities. The test helps:<br/>

•	Measure knowledge across many fields<br/>
•	Compare different AI models<br/>
•	Find where models are strong or weak<br/>
•	Track AI progress in understanding complex topics

Run (using stored dataset): 
```
C:\Windows\py.exe .\MMLU.py
```
Run (using HuggingFace dataset):  
```
C:\Windows\py.exe .\MMLU_Eval_HF.py
```

### BLEU Evaluation<br/>
BLEU (Bilingual Evaluation Understudy) is a metric that measures machine translation quality by comparing AI translations to human translations. It works by counting matching words and phrases, giving a score from 0 to 1 (or 0-100%). Higher scores mean the AI translation is closer to human translation.

Run (using stored dataset):
```
C:\Windows\py.exe .\BLEU_Eval.py --input translation-test-data.json --output translation_results.json
```
Run (using HuggingFace dataset):  
```
C:\Windows\py.exe .\BLEU_Eval_HF.py
```

### HellaSwag Evaluation<br/>
HellaSwag is an AI test that measures common sense understanding. It shows the AI the start of a situation and asks it to pick the most logical ending from multiple choices. It helps evaluate if AI can understand everyday situations and predict what would naturally happen next. 

Run (using stored dataset):  
```
C:\Windows\py.exe .\HellaSwag.py
```
Run (using HuggingFace dataset):  
```
C:\Windows\py.exe .\HellaSwag_HF.py
```

### TruthfulQA Evaluation<br/>
TruthfulQA is a benchmark that tests whether AI models give accurate answers or repeat common misconceptions. It uses ~800 questions across topics like health and history, where each question has both a true answer and false answers reflecting popular myths. Models are scored on both truthfulness and informativeness, with evaluation done through both free-form answers and multiple choice questions.

Run (using stored dataset):  
```
C:\Windows\py.exe .\TruthfulQA.py<br/>
```
Run (using HuggingFace dataset):  
```
C:\Windows\py.exe .\TruthfulQA_HF.py
```

### HumanEval Evaluation<br/>
HumanEval is a coding benchmark with 164 Python programming problems. Each includes a function description and test cases. Models must generate working code solutions, and are evaluated on pass@k - the chance of getting a correct solution when sampling k times. It tests actual code execution rather than just comparing to reference solutions.

Run (using stored dataset): 
```
C:\Windows\py.exe .\HumanEval.py
```
Run (using HuggingFace dataset):  
```
C:\Windows\py.exe .\HumanEval_HF.py
```

### Rouge Evaluation<br/>
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a metric for evaluating text summarization and translation by comparing machine-generated text against human references. It counts matching elements (like words or phrases) between them, measuring both how much of the reference is captured (recall) and how accurate the generated text is (precision).

The following variants are evaluated:<br/>

•	ROUGE-N: Measures overlapping n-grams (e.g., ROUGE-1 for single words, ROUGE-2 for word pairs)<br/>
•	ROUGE-L: Uses longest common subsequence to capture sentence-level structure

Run (using stored dataset):  
```
C:\Windows\py.exe .\ROUGE_Eval.py
```
Run (using HuggingFace dataset):  
```
C:\Windows\py.exe .\ROUGE_Eval_HF.py
```

### Bert Evaluation<br/>
BERT evaluation typically uses benchmarks like GLUE and SQuAD to test the model's language understanding abilities across tasks like question answering, sentiment analysis, and text similarity. Performance is measured using accuracy and F1 scores.

Run (using stored dataset):  
```
C:\Windows\py.exe .\BERT_Eval.py
```
Run (using HuggingFace dataset):  
```
C:\Windows\py.exe .\BERT_Eval_HF.py
```

### Benchmark scores:
Benchmark scores based on the stored data file:
• MMLU: Perfect accuracy (100%) on 30 samples
• HellaSwag: Perfect accuracy (100%) on 10 samples
• BLEU: Strong performance with 88% score on 11 samples
• BERT metrics show some concerning results with negative precision (-0.052) and low F1 score (0.099)
• Rouge scores show moderate performance across Rouge-1, Rouge-2, and Rouge-L metrics
• TruthfulQA shows a perfect truthful rate (1.0) but a low average similarity score (0.135)

Benchmark scores based on the HuggingFace dataset:
• MMLU: Lower accuracy (60%) on 10 samples
• BLEU: Significantly lower score (31%) on 15 samples
• BERT shows much better performance with balanced precision (0.82) and recall (0.88)
• Rouge scores generally show higher precision but lower recall compared to the stored dataset
• TruthfulQA shows very low accuracy (3.4%) on a much larger sample size (817 samples)

Some important observations:
1. The sample sizes vary significantly between stored data files and HuggingFace dataset
2. There are some missing data points (e.g., HellaSwag for HuggingFace dataset)
3. The dramatically different results between datasets suggest there might be important underlying differences in how the tests were conducted or what data was used

Rather than drawing definitive conclusions about the model's performance, I'd recommend:
• Investigating why there are such large disparities between the datasets
• Understanding the specific test conditions and data preprocessing methods used
• Conducting additional tests with larger, consistent sample sizes
• Looking into the negative precision value in the BERT test, as this seems unusual 

## Overfitting and Underfitting Evaluation<br/>
Overfitting happens when a model learns the training data too well, including noise, leading to high training accuracy but poor performance on new data. It shows a large gap between training and validation metrics. Underfitting occurs when a model is too simple to capture the data's patterns, resulting in poor performance on both training and validation datasets.

Key Indicators:<br/>
Overfitting: High training accuracy, low validation accuracy.<br/>
Underfitting: Low accuracy on both training and validation data.

Mitigation:<br/>
Overfitting: Simplify the model, use regularization, or gather more data.<br/>
Underfitting: Increase model complexity, train longer, or optimize features.

## Automated Tests<br/>
Functional Automated tests are stored inside `Model_Evaluation` folder. Test script is written in Python. Functional tests cover:<br/>

•	Functional completeness with unseen data<br/>
•	Temperature testing<br/>
•	Prompt testing<br/>
•	Chain of thought prompts<br/>
•	Does it stay relevant to the topic?<br/>
•	Fantasy claims definition<br/>
•	Repeatability tests<br/>
•	Ask questions in different phrases<br/>
•	Style transfer testing<br/>
•	Intent recognition<br/>
•	Context management testing<br/>
•	User action prompts with options<br/>
•	Responsible testing<br/>
•	Fairness testing<br/>
•	Bias detection and mitigation<br/>
•	Transparency testing<br/>
•	Ethical testing<br/>
•	Data privacy and security testing

## API Tests<br/>
API tests are written in Postman. They aim to validate:<br/>

•	Response accuracy and quality<br/>
•	Input/output format validation (schema validation)<br/>
•	Status code<br/>
•	Response times and correctness<br/>
•	Authentication/authorization<br/>
•	Request/response headers<br/>
•	Data formats (JSON, streaming)<br/>
•	Very long/short inputs<br/>
•	Special characters<br/>
•	Different languages<br/>
•	API key validation<br/>
•	Data privacy compliance<br/>

## Performance Tests<br/>
Postman collection should be imported to JMeter.<br/>

JMeter configuration:<br/>

• Create JMeter project or user existing one<br/> 
• Add new Thread Group<br/>
• Add a Recording controller (Thread Group > Add > Logic Controller > Recording Controller)<br/>
• Add HTTP(S) Test Script Recorder (Test plan > Add > Non-Test Elements > HTTP(S) Test Script Recorder<br/>

Postman configuration:<br/>

• Select Settings > Proxy > Use custom proxy configuration option<br/>
• Fill Proxy server fields: enter your IP and port:8888 (this is JMeter port)<br/>

Action:<br/>

• Collection should be open in Test Runner<br/>
• From Run oder in Postman, select those test you want to run<br/>
• In JMeter, inside HTTP(S) Test Script Recorder press Start button<br/>
• On Reccorder:Transaction Control, in Transaction name field enter the name of tests you previously selected to run in Postman<br/>
• Return to Postman and press Run to execute tests you selected from Collection<br/>
• In JMeter you should see all executed tests inside Recording Controller<br/>

Performance tests are executed to measure how an AI model performs under high demand including:<br/>

•	Testing with many simultaneous users or requests<br/>
•	How fast does the system respond under load<br/>
•	CPU, memory, and network usage at different load levels<br/>
•	Maximum load before performance degrades/fails<br/>

## Use Case: Testing of AI Model<br/>
•	Test the AI model using semantically equivalent sentences and create an automated library for this purpose<br/>
•	Maintain configurations of basic and advanced semantically equivalent sentences with formal and informal tones and complex words<br/>
•	Automate end-to-end scenario (requesting AI model, getting a response and validating the response action with accepted output)<br/>
•	Generate automated scripts in Python for execution


