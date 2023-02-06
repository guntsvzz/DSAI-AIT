```
Paper : Thai Nested Named Entity Recognition Corpus
Link : https://aclanthology.org/2022.findings-acl.116/
Github : https://github.com/vistec-AI/Thai-NNER
Venue : ACL
```

| Topic        | Thai Nested Named Entity Recognition Corpus |
|--------------|---------------------------------------------|
| Question     | N-NER benefits to downstream tasks; however, low-resources languages, such as Thailand, cannot be reliable to NNER models |
| Related Work | NNE, GENIA, ACE-2005 (English) ; NoSta-D (German) ; VLSP-2018 (Vietnamese)                             |
| Solution     | Own combination dataset from new articles ([Prachathai](https://huggingface.co/datasets/prachathai67k)) and resturant reviews ([Wongnai](https://github.com/wongnai/wongnai-corpus)) following CoNLL format|
| Method       | 264,798 mentions organized into 104 classes and has a maximum depth of 8 in order to overlook fine-grained class and coarse-grained |
| Result       | 1. XLM-R models performances are better than WangchanBERTa (monolingual)  <br />2. F1 score of Pyramid model on the NNE corpus is 94.68, whereas is only 78.50 of Thai NNER dataset |
| Conclusion   | overall performance, it should pay attention to recall |
| Limitation | 1.Incorrect span prediction  <br />2. Ambiguous entity mentions  <br />3. Ambiguity between fine-grained classes  <br />4. Scarcity of training samples |