```
Paper : Enriching Word Vectors with Subword Information
Link : https://arxiv.org/abs/1607.04606
Venue: ACL
```

| Topic        | Enriching Word Vectors with Subword Information        |
|--------------|--------------------------------------------------------|
| Question     | How to improve out-of-vocabulary (OOV) |
| Related Work | morphological information, character n-gram representation|
| Solution     | character-based : subword information adding "<" and ">" to perform prefix and suffix |
| Method       | considering subword units, and representing words by a sum of its chatacter n-grams|
| Result       | fast and capture OOV, 3-6-gram is optimal|
| Conclusion   | subword representations relying on morphological analysis|
| Limitation   | 1. high memory requirement such WHERE with 3-gram: <wh, whe, her, ere, re> such subword is not necessary <br> 2. It doesn't consider context word|