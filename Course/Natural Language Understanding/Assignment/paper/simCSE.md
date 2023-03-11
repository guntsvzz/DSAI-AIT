```
Paper : SimCSE: Simple Contrastive Learning of Sentence Embeddings 
Link : https://aclanthology.org/2021.emnlp-main.552/
Venue: ACL
```

| Topic        | SimCSE: Simple Contrastive Learning of Sentence Embeddings  |
|--------------|----------------------------------------------------------------|
| Question     | learning universal sentence embedding is struggle|
| Related Work | Supervised sentence embedding is stronger performance|
| Solution     | passs the same sentence twice which are nomral sentence embedding and another with dropout in BERT & RoBERTa|
| Method       | 1. Unsupervised SimCSE : dropout masks placed on fc layer as well as attention probabilities</br> 2. Supervised SimCSE : finetuneed on NLI dataset|
| Result       | 1. dropout act as data augmentation</br> 2 </br>|
| Conclusion   | Dropout noise utilizes Unsupervised and Supervised approach|
| Limitation   | it may not be able to capture the semantic meaning of the sentences|