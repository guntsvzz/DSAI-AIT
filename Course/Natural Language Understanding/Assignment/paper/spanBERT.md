```
Paper : SpanBERT: Improving Pre-training by Representing and Predicting Spans
Link : https://arxiv.org/abs/1907.10529
Venue: TACL 
```

| Topic        | SpanBERT: Improving Pre-training by Representing and Predicting Spans |
|--------------|--------------------------------------------------------------|
| Question     | many NLP task involve reasoning about relationship between two or more span of text|
| Related Work | classical BERT: MLM & NSP|
| Solution     | mask random contiguous spans & span-boundary objective (SBO)|
| Method       | 1. Masking contiguous random spans, rather than random tokens </br> 2. training the span boundary representaion to predict the entire content of the masked span, without relying on the individual tokens|
| Result       | 1. SpanBERT outperform BERT on almost every task</br>2. SpanBERT is better at extractive question answering 3.bi-sequence training with NSP objective drop performance on downsteam tasks</br>|
| Conclusion   | outperform all BERT baselines on a variety of task|
| Limitation   | SpanBERT requires more computational resources than BERT in order to more robust from masked span|