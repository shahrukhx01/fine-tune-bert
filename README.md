# Fine Tune Bert for QUESTION CLASSIFICATION 

| Train Loss    | Validation Acc.| Test Acc.|
| ------------- |:-------------: | -----:   |
| 0.000806      | 0.99  | 0.992    |

# USAGE
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
  
tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/bert-mini-finetune-question-detection")

model = AutoModelForSequenceClassification.from_pretrained("shahrukhx01/bert-mini-finetune-question-detection")
```
Trained to add feature of Question vs Statement classification in (Haystack)[https://github.com/deepset-ai/haystack/issues/611]

Problem Statement:
One common challenge that we saw in deployments: We need to distinguish between real questions and keyword queries that come in. We only want to route questions to the Reader branch in order to maximize the accuracy of results and minimize computation efforts/costs.

Describe the solution you'd like

New class QueryClassifier that takes a query as input and determines if it is a question or a keyword query.
We could start with a very basic version (maybe even rule-based) here and later extend it to use a classification model.
The run method would need to return query, "output_1" for a question and query, "output_2" for a keyword query in order to allow branching in the DAG.

Describe alternatives you've considered
Later it might also make sense to distinguish into more types (e.g. full sentence but not a question)

Additional context
We could use it like this in a pipeline

Baseline:
https://www.kaggle.com/shahrukhkhan/question-v-statement-detection

Dataset:
https://www.kaggle.com/stefanondisponibile/quora-question-keyword-pairs

Kaggle Notebook:
https://www.kaggle.com/shahrukhkhan/question-vs-statement-classification-mini-bert/
