```
Paper : Quality Controlled Paraphrase Generation
Link : https://aclanthology.org/2022.acl-long.45/
Venue: ACL
```

| Topic        | Quality Controlled Paraphrase Generation               |
|--------------|--------------------------------------------------------|
| Question     | producing diversity is the main challenging to control paraphase geneation|
| Related Work | 1.Hamming distance </br> 2.exposing control mechanisms that are
manipulated to produce either lexically or syntactically</br> 3. constituency tree as the syntax </br> 4. reinforcement
learning with quality-oriented reward |
| Solution     | QCPG, a quality-guided controlled paraphrase generation model which present by 3-dimension vector of semantic similairity, and syntactic and lexical distances|
| Method       | 1. Quantifying Paraphrase Quality </br>- syntatic score normalized tree edit distance</br>- lexcical normalzed character-level minimum edit distnace btw bag of words </br> - semantic score normalized using the sigmoid function </br> 2. the control values should be determined with respect to the input sentence |
| Result       | 1. increasing the input offset gobe good control mechanism</br>2.manipulating the input offset control o to meet her desired quality values</br>3. |
| Conclusion   | |
| Limitation   | |