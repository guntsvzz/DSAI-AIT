# Your first reading assignment
```
As we have mentioned in the class, read just one ACL / EMNLP / NIPS paper.   You can select any paper you like.

Append a table in a Github repo.

Since this is your first week, we shall warm up and give you two weeks for your first paper.

PS:  Note that you may not completely understand everything, but just try your best.
   
Once you reach the mark of 50+ papers, you will start to understand and become superman.

Enjoy reading.

Point criteria:
0:  Not done / copy
1:  Ok
2:  With good, clear, precise, personal explanations in own words
```

```
Paper : Thai Nested Named Entity Recognition Corpus
Link : https://aclanthology.org/2022.findings-acl.116/
Github : https://github.com/vistec-AI/Thai-NNER
```

| Topic        | Thai Nested Named Entity Recognition Corpus                                                            |
|--------------|--------------------------------------------------------------------------------------------------------|
| Question     | N-NER benefits to downstream tasks; however, low-resources languages cannot be reliable to NNER models |
| Related Work | NNE, GENIA, ACE-2005 (English) ; NoSta-D (German) ; VLSP-2018 (Vietnamese)                             |
| Solution     | Own dataset from new articles ([Prachathai](https://huggingface.co/datasets/prachathai67k)) and resturant reviews ([Wongnai](https://github.com/wongnai/wongnai-corpus))                |
| Method       | 264,798 mentions organized into 104 classes and has a maximum depth of 8 layers                                                                                                       |
| Result       | 1. XLM-R models performances are better than WangchanBERTa (monolingual) 2. Pyramid model on the NNE corpus is 94.68, whereas is only 78.50 of Thai NNER dataset
                                                                                                     |
| Conclusion   |                                                                                                        |
| Future Work  |                                                                                                        |