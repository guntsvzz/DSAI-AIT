```
Paper : Improving Language Understanding by Generative Pre-Training
Link : https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
Venue: OPENAI 
```

| Topic        | Improving Language Understanding by Generative Pre-Training  |
|--------------|--------------------------------------------------------------|
| Question     | labeled data is not enough to learning all specific tasks    |
| Related Work | 1. Semi-upervised Learning </br> 2. Unsuperversied pre-training </br> 3. Auxiliary training objective </br>|
| Solution     | combination of unseupervised pre-training and supervised fine-tuning|
| Method       | 1. use LM objectives o nthe unlabeled data to learn the initial parameters of model</br> 2. its parameters are modified based on the supervised objective of the task through fine-tuning</br>|
| Result       | 1. large datset get benefit from auxiliary objective </br> 2. with out pre-training, it drop performance across all the tasks  </br>|
| Conclusion   | 1. pre-training on a diverse corpus is significant </br> 2. Unsuperversied traiing boost performance on discriminative </br>|
| Limitation   | Due to training on a wide variety of text, it may not perform as well as other models that have been fine-tuned on specific tasks.|