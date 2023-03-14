```
Paper : BART: Denoising Sequence-to-Sequence Pre-training for Natural
Language Generation, Translation, and Comprehension
Link : https://aclanthology.org/2020.acl-main.703/
Venue: ACL
```

| Topic        | BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension|
|--------------|----------------------------------------------------------------|
| Question     | MLM focus on particular types of end task such as span prediction, generation|
| Related Work | BERT, UniLM, MASS, XL-Net, GPT|
| Solution     | combination Bidirectional (BERT) and Auto-Regressive (GPT) Transformers|
| Method       | Pretraining has 2 stages : text is corrupted with an arbitrary noiseing function and a seq2seq model is learned to construct the original text |
| Result       | 1.Token deletion, masling, self-attention mask is crucial </br> |
| Conclusion   | learn to map corrupted documents to the original|
| Limitation   | Perform well only Transaltion and comprehension |