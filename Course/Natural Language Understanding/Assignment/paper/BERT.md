```
Paper : BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Link : https://arxiv.org/pdf/1810.04805.pdf
Venue: Google 
```

| Topic        | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  |
|--------------|--------------------------------------------------------------|
| Question     | |
| Related Work | 1. ELMo add additional feature which is a task-specific</br> 2. GPT-1, a fine-tuing all pretrained parameters|
| Solution     | pre-train deep bidirectional representations from unlabeled text by jointly confiitoning on both left and right context in all layers.|
| Method       | 1. Masked Language Model (MLM) randomly mask some of the tokens from the input and the objective is to predict the original vocabulary </br> 2. Next Sentence Prediction (NSP) jointly pre-trains text-pair representations|
| Result       | 1. Large BERT is unstable on small datasets </br> 2. removing NSP drop performance on QNLI, MNLI and SQuAD </br> 3. MLM model is better than LTR (Left-to-Right) model on MRPC and SQuAD </br> 4. increaing the model size will lead to improve on large-scale task |
| Conclusion   | integrating many language system result in enabling even low-resource task|
| Limitation   | unsuitable for various pair regression tasks because of too many possible combinations|