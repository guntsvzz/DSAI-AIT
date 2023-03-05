```
Paper : RoBERTa: A Robustly Optimized BERT Pretraining Approach
Link : https://arxiv.org/pdf/1907.11692.pdf
Venue: Facebook AI 
```

| Topic        | RoBERTa: A Robustly Optimized BERT Pretraining Approach  |
|--------------|--------------------------------------------------------------|
| Question     | Triaining is expensive which mostly perform on provate dataset of different size| 
| Related Work | 1. self-triaing methods : BERT, GPT-1 </br> 2. Large Batch size of BERT</br>|
| Solution     | measure impact of the key hyperparameter and training data size|
| Method       | 1. training the model longer, bigger batches</br> 2. removing NSP</br> 3. training on longer sequence</br> 4. chagning the masking pattern applied to the training data </br>|
| Result       | 1. removing the NSP loss improves downsteam tasks performance </br> 2. using individual sentence drop on downstream taks performance </br> 3. it depends only on single-task finetuing</br>|
| Conclusion   | RoBERTa achieves SOTA results on GLUE, RACE and SQuAD|
| Limitation   | Due to Large Memory computation, it is practical to low resource. |