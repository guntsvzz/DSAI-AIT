```
Paper : Neural Machine Translation by Jointly Learning to Align and Translate
Link : https://arxiv.org/abs/1409.0473
Venue: ICLR 
```

| Topic        | Neural Machine Translation by Jointly Learning to Align and Translate  |
|--------------|------------------------------------------------------------------------|
| Question     | it difficult for NN to cope with long sentences |
| Related Work | 1. mixtue of Gaussian kernel to compute the weight<br> 2. condtional probability of a word given a fixed number of the preceding words<br> 3. addtional feature in the phrase-based SMT system|
| Solution     | extend align and translate jointly|
| Method       | 1. adding NN components to existing translation <br> 2. taking a weighted sum of all the annotation<br>|
| Result       | 1. soft-aligment deal with source and target phrases of different length <br> 2. it's not required encoding a long sentence in to a fixed-length vector<br>|
| Conclusion   | model can correctly align each target with their annotation in the source sentence|
| Limitation   | unable to handle unkown or rare words.|