```
Paper : Unsupervised Syntactically Controlled Paraphrase Generation with
Abstract Meaning Representations
Link : https://aclanthology.org/2022.findings-emnlp.111.pdf
Venue: ACL 
```

| Topic        | Unsupervised Syntactically Controlled Paraphrase Generation with Abstract Meaning Representations |
|--------------|---------------------------------------------------------------------------------------------------|
| Question     | Non-parallel pair still suffer from relatively poor performance. Mostly target sentence attempt to copy source sentence too much|
| Related Work | Order Control (REAP-2020), Syntactic Control (SCPN-2018, SynPG-2021), Abstract meaning representation (AMR)|
| Solution     | adding AMR graph to disentangle semantic embedddings|
| Method       | 1. Semantic Embedding use a pre-trained AMR parser to get AMR graph</br> 2. Syntatic Embedding as a constituency parse</br> 3. Decoder learn from those two embedding reconstruct with CE loss |
| Result       | using AMR can learn beter disentangled embeddings and capture semantics better </br>|
| Conclusion   | AMRPG captures semantics better and generate more accurate than previous SOTA|
| Limitation   | 1. using full constitiuency parse is required addiational efforts </br> 2. requiring a pre-trained AMR parser which is costly in training|