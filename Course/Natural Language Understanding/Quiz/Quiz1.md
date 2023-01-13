## Quiz 1 : Word2Vec

1. Word2vec does not work well with (Out of Vocabulary) OOV. What is one possible way to solve this?
    - Use character based or sub-words embedding
2. What is the idea of skip gram?
    - Predicting the context word using a center word
3. What is the idea about CBOW?
    - Predicting the center word using the context words
4. What is the idea of negative sampling?
    - Picking some negative samples and make the difference between positive and negative samples to be big.
    - Negative sampling is efficient.
5. Word2vec is doubtful whether contextual information is fully captured. What is one possible way to solve this?
    - Pass these trained embeddings through some LSTM, and get the resulting encodings as embeddings
6. Word2vec only looks at local words. What is one possible way to solve this
    - Use co-occurrence statistics
7. What is true about word2vec?
    - W2V is a effective method of creating word embeddings
    - W2V can create word embeddings based on their occurrences in the text
8. What is true about negative sampling?
    - Maximize the probability of real outside word appears, and minimize the probability that random words appear around the center word
    - Negative sampling is a widely used technique in the deep learning field