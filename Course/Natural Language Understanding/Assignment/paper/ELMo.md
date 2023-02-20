```
Paper : Deep contextualized word representations
Link : https://aclanthology.org/N18-1202/
Venue: ACL
```

| Topic        | Deep contextualized word representations                |
|--------------|---------------------------------------------------------|
| Question     | Due to word/character-based cannot capture context representation then how to fix it |
| Related Work | 1.unbeled data in sequence tagging model <br> 2.bidirectional RNN model to predict future word|
| Solution     | extract embeddding from bidirectional LM|
| Method       | 1.use Word and LM embedding to predict tagging task (NER or chucking) <br> 2. pretrained the Forward and Backward LM<br> 3. concatenate LM with sequence model<br>|
| Result       | 1. adding LM help composition functions (RNN) <br> 2. increasing parameter doesn't improve performance|
| Conclusion   | adding Backward LM improves performance|
| Limitation   | 1. relying on both labeled and unlabeled data as well as larger trining sets<br> 2. high computuational as bidirectional RNN or GRU 3|