```
Paper : WangchanBERTa Pretraining transformer-based Thai Language Models
Link : https://arxiv.org/abs/2101.09635
Github : https://github.com/vistec-AI/thai2transformers
Venue: NAACL 
```

| Topic        | WangchanBERTa Pretraining transformer-based Thai Language Models |
|--------------|------------------------------------------------------------------|
| Question     | large-scale multi-lingual pretraining doesn't consider language specific feature for low-resouce such as Thai|
| Related Work | 1. preprocessing tokenizer family : SentencePiece tokenizer, Word-level tokenizer, Syllable-level tokenize, SEFR tokenizer<br>2. RoBERTa-base Achitecture|
| Solution     | pretraining with Thai resources which are Assorted Thai Texts Dataset and Wikipedia-only Dataset|
| Method       | replace HTML forms, empty brackets, repetitive character such as ดีมากกกกก to ดีมาก then use 4 different tokenizers |
| Result       | 1. WangchanBERTa outperforms baselines (NBSVM, CRF and ULMFit) and multi-lingual models (XLMR and mBERT <br>|
| Conclusion   | 1. a multi-lingual model (XLMR) is better when it include multi-lingual elements namely the English-to-Thai translated texts <br> 2. There are no difference performance for sequence/token classification tasks. |
| Limitation   | bias-measuring datasets in Thai contexts |
