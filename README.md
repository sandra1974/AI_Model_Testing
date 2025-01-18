# AI_Model_Testing
This project aims to validate how an AI model behaves in different scenarios and whether it returns exact, accurate, and relevant answers. It is also tested how the AI model behaves when many users simultaneously ask questions and what is the speed of the AI model response. All these tests are conducted on AI model claude-3-5-sonnet-20241022. Test scripts are divided into two separate folders: 

•	Model_Engineering<br/> 
•	Model_Evaluation

During the model engineering phase, benchmark tests and overfitting and underfitting evaluations are completed, while functional, performance, and API tests are executed in the model evaluation phase. However, API tests can also be performed at an earlier stage.

## Prerequisites
The following environment should be set up:<br/>
•	Visual Studio<br/>
•	Python<br/>
•	Postman<br/>
•	JMeter<br/>
•	API key generated from this link: https://console.anthropic.com/dashboard<br/> 
•	Configure environment variables > > > variable name: $ANTHROPIC_API_KEY and variable value: your api key<br/>

##Benchmark Testing<br/>
Benchmark tests are written in Python and completed in 2 ways:<br/>
-	Using dataset files stored in Dataset folder<br/>
-	Using the HuggingFace dataset<br/> 

All Benchmark tests are stored inside Model_Engineering folder.<br/>

###MMLU evaluation<br/>
MMLU (Massive Multitask Language Understanding) is a testing tool that evaluates how well AI language models perform across 57 different subjects. It works like a comprehensive exam, using multiple-choice questions to test knowledge in areas from basic science to advanced professional topics like law and medicine. Its main purpose is to measure how well AI models understand and can reason about different subjects, and to provide a way to compare different AI models' capabilities. The test helps:
•	Measure knowledge across many fields
•	Compare different AI models
•	Find where models are strong or weak
•	Track AI progress in understanding complex topics

Run (using stored dataset):  C:\Windows\py.exe .\MMLU.py
Run (using HuggingFace dataset):  C:\Windows\py.exe .\MMLU_Eval_HF.py

###BLEU evaluation
BLEU (Bilingual Evaluation Understudy) is a metric that measures machine translation quality by comparing AI translations to human translations. It works by counting matching words and phrases, giving a score from 0 to 1 (or 0-100%). Higher scores mean the AI translation is closer to human translation.

Run (using stored dataset):
C:\Windows\py.exe .\BLEU_Eval.py --input translation-test-data.json --output results.json
Run (using HuggingFace dataset):  C:\Windows\py.exe .\BLEU_Eval_HF.py

###HellaSwag evaluation
HellaSwag is an AI test that measures common sense understanding. It shows the AI the start of a situation and asks it to pick the most logical ending from multiple choices. It helps evaluate if AI can understand everyday situations and predict what would naturally happen next. 

Run (using stored dataset):  C:\Windows\py.exe .\ HellaSwag.py 
Run (using HuggingFace dataset):  C:\Windows\py.exe .\HellaSwag_HF.py

###TruthfulQA evaluation
TruthfulQA is a benchmark that tests whether AI models give accurate answers or repeat common misconceptions. It uses ~800 questions across topics like health and history, where each question has both a true answer and false answers reflecting popular myths. Models are scored on both truthfulness and informativeness, with evaluation done through both free-form answers and multiple choice questions.

Run (using stored dataset):  C:\Windows\py.exe .\ TruthfulQA.py
Run (using HuggingFace dataset):  C:\Windows\py.exe .\ TruthfulQA_HF.py

###HumanEval evaluation
HumanEval is a coding benchmark with 164 Python programming problems. Each includes a function description and test cases. Models must generate working code solutions, and are evaluated on pass@k - the chance of getting a correct solution when sampling k times. It tests actual code execution rather than just comparing to reference solutions.

Run (using stored dataset):  C:\Windows\py.exe .\ HumanEval.py
Run (using HuggingFace dataset):  C:\Windows\py.exe .\ HumanEval_HF.py

###Rouge evaluation
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a metric for evaluating text summarization and translation by comparing machine-generated text against human references. It counts matching elements (like words or phrases) between them, measuring both how much of the reference is captured (recall) and how accurate the generated text is (precision).

The following variants are evaluated:
•	ROUGE-N: Measures overlapping n-grams (e.g., ROUGE-1 for single words, ROUGE-2 for word pairs)
•	ROUGE-L: Uses longest common subsequence to capture sentence-level structure

Run (using stored dataset):  C:\Windows\py.exe .\ ROUGE_Eval.py
Run (using HuggingFace dataset):  C:\Windows\py.exe .\ ROUGE_Eval_HF.py

###Bert evaluation
BERT evaluation typically uses benchmarks like GLUE and SQuAD to test the model's language understanding abilities across tasks like question answering, sentiment analysis, and text similarity. Performance is measured using accuracy and F1 scores.

Run (using stored dataset):  C:\Windows\py.exe .\ BERT_Eval.py
Run (using HuggingFace dataset):  C:\Windows\py.exe .\ BERT_Eval_HF.py

##Overfitting and Underfitting Evaluation
Overfitting happens when a model learns the training data too well, including noise, leading to high training accuracy but poor performance on new data. It shows a large gap between training and validation metrics. Underfitting occurs when a model is too simple to capture the data's patterns, resulting in poor performance on both training and validation datasets.

Key Indicators:
Overfitting: High training accuracy, low validation accuracy.
Underfitting: Low accuracy on both training and validation data.

Mitigation:
Overfitting: Simplify the model, use regularization, or gather more data.
Underfitting: Increase model complexity, train longer, or optimize features.

##Automated Tests
Functional Automated tests are stored inside Model_Evaluation folder. Test script is written in Python. Functional tests cover:
•	Functional completeness with unseen data
•	Temperature testing
•	Prompt testing
•	Chain of thought prompts
•	Does it stay relevant to the topic?
•	Fantasy claims definition
•	Repeatability tests
•	Ask questions in different phrases
•	Style transfer testing
•	Intent recognition
•	Context management testing
•	User action prompts with options
•	Responsible testing
•	Fairness testing
•	Bias detection and mitigation
•	Transparency testing 
•	Ethical testing
•	Data privacy and security testing

##API Tests
API tests are written in Postman. They aim to validate:
•	Response accuracy and quality
•	Input/output format validation (schema validation)
•	Status code
•	Response times and correctness
•	Authentication/authorization
•	Request/response headers
•	Data formats (JSON, streaming)
•	Very long/short inputs
•	Special characters
•	Different languages
•	API key validation
•	Data privacy compliance

##Performance Tests
Postman collection should be imported to Postman.
Performance tests are executed to measure how an AI model performs under high demand including:
•	Testing with many simultaneous users or requests
•	How fast does the system respond under load
•	CPU, memory, and network usage at different load levels
•	Maximum load before performance degrades/fails

##Use Case: Testing of AI model
•	Test the AI model using semantically equivalent sentences and create an automated library for this purpose
•	Maintain configurations of basic and advanced semantically equivalent sentences with formal and informal tones and complex words
•	Automate end-to-end scenario (requesting AI model, getting a response and validating the response action with accepted output)
•	Generate automated scripts in Python for execution


