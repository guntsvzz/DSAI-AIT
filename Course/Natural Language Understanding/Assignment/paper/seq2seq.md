```
Paper : Sequence to Sequence Learning with Neural Networks
Link : https://arxiv.org/abs/1409.3215
Venue: 
```

| Topic        | Sequence to Sequence Learning with Neural Networks      |
|--------------|---------------------------------------------------------|
| Question     | 1. DNN cannot map sequence to sequence <br> 2. RNN difficult to train with long term dependencies <br> |
| Related Work | mapping the entire input sentence to vector|
| Solution     | use one LSTM to read the input sentence (Decoder) and another LSTM extract the output sentecne (Encoder) |
| Method       | estimate the conditional probability using softmax then Decoder give last hidden state to Encoder |
| Result       | 1. information of long sentence is not lost in LSTM <br> 2. reverseing the sequence extend improvement <br> 3. representations are effect to order of words <br>|
| Conclusion   | LSTM-based on limited vocabulary, can outperform SMT-based which vocabury is unlimited|
| Limitation   | train on reversed dataset difficult translating long sentences|