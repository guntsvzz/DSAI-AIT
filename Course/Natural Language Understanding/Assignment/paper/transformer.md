```
Paper : Attention Is All You Need
Link : https://arxiv.org/abs/1706.03762
Venue: NIPS
```

| Topic        | Attention Is All You Need |
|--------------|---------------------------|
| Question     | recurrence have a problem a critical at longer sequence lengths, as memory constraints limit batching across examples.|
| Related Work | Attention mechanism|
| Solution     | reducing sequential computattion by discaring reccurrence layers and use entirely attention mechanism allowing to more parallelization|
| Method       | 6 stack identical layer <br> 1. Dncoder : multi-head & feedforward with layer norm as sub-layers<br> 2. Encoder : modifly the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent position<br> |
| Result       | 1.model achieves a BLEU score of 41.0 <br> 2. 1/4 the training cost of the previous SOTA (sequential operations and maximum path length is O(1)) <br>|
| Conclusion   | encoder-decoder architecture wit hmulti-headed self-attention|
| Limitation   | When it doesn't have RNN making generation less sequential|