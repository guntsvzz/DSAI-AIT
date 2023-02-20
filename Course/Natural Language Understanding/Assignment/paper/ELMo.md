```
Paper : Deep contextualized word representations
Link : https://aclanthology.org/N18-1202/
Venue: ACL
```

| Topic        | Deep contextualized word representations                |
|--------------|---------------------------------------------------------|
| Question     | how generating word embedding that capture the context representation|
| Related Work | Subword information, Unsupervised language model (TagLM)|
| Solution     | combination of the intermediate layer representations in the biLM |
| Method       | 1. 2biLSTM layer : lower-level for syntaax and higher-layer for semantics<br> 2. Use weight average instead of last hidden state<br> 3. Freeze weight of ELMo<br> 4. Concatenating into intermediate layers same as TagLM<br>|
| Result       | |
| Conclusion   | Adding ELMp improve NLP many tasks|
| Limitation   | Adding ELMo in small training sets is not different slightly much comparing with baseline|