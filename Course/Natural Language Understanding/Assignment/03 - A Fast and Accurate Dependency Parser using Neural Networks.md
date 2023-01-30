```
Paper : A Fast and Accurate Dependency Parser using Neural Networks
Link : https://aclanthology.org/D14-1082.pdf
```

| Topic        | A Fast and Accurate Dependency Parser using Neural Networks |
|--------------|--------------------------------------------------------------------------------------------------------|
| Question     | sparse feature is poor generealization and costly computational parsing speed.|
| Related Work |  1. localist one-hot word representations (1999) <br /> 2. a shift reduce constituency parser with one-hot word representations and subsequent parsing work (2005) <br /> 3. a simple synchrony network (2007) parsing work <br /> 4. Incremental Sigmoid Belief Network (2007) <br /> 5. a Temporal Restricted Boltzman Machine (2011) <br /> 6. a Temporal Restricted Boltzman Machine (2013) |
| Solution     | propose transition-based dependency parser using dense feature instead. |
| Method       | input layer contain : words, POS tages, and arc labels |
| Result       | 1. transition-based perform well in LAS espeacially fine-grained label set  <br />2. POS tags' embeddings is the important feature due to capture most of the label information |
| Conclusion   |  outperform accuracy and speed than other greedy parsers that using sparse features. |
| Limitation   |  only rely on dense features (POS tags and arc labels(dependency))|