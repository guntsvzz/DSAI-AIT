```
Paper : Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
Link : https://arxiv.org/abs/1908.10084
Venue: EMNLP 
```

| Topic        | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks |
|--------------|----------------------------------------------------------------|
| Question     | BERT and RoBERTa still unstable in pair regression tasks because of huge possibel combinations. Moreover, time computation still high.|
| Related Work | 1.Skip-Thought trains an encoder-decoder architecture to predict hhe surroudning sentences. </br> 2. InferSent uses labeled data to train siamese BiLSTM with max-pooling over the input|
| Solution     | modification BERT network using Siamese and triplet network to derive sentence embeddings that compared using cosine-simmilarity |
| Method       | 1. Siamese network fixed-sized vectors for input sentences then using similarity measure. </br> 2. Adding a pooling to the output of BERT/RoBERTa</br> 3. find-tuned SBERT on NLI data. </br> 3.|
| Result       | 1. SBERT reduce 65 hours computation from BERT/RoBERTa to be 5 seconds</br> 2.it's not benefitcal for the Q&A task|
| Conclusion   | 1. sentence embedding from SBERT capture well sentiment information </br> 2. BERT or RoBERTa is slightly different to improve in experiment|
| Limitation   | it still struggle when dataset providing less topics which model cannot perform STS taks.|