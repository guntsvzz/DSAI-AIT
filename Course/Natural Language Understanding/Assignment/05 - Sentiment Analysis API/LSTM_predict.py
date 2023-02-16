from LSTM_model import *
import torch
import pytreebank
import spacy
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd

import matplotlib.pyplot as plt
import base64
import io

#Load GPU
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

####Model Part
#Load Data
train = pytreebank.import_tree_corpus("./sentiment/train.txt")
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

def seperate(dataset): #Use All nodes
    seperation = []
    for data in dataset:
        for label, text in data.to_labeled_lines():
            seperation.append((label,text))
    return seperation

train_sep = seperate(train)
text_pipeline = lambda x: vocab(tokenizer(x))

def yield_tokens(data_iter): #data_iter, e.g., train
    for _, text in data_iter: 
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_sep), specials=['<unk>','<pad>','<bos>','<eos>'], special_first = True)
vocab.set_default_index(vocab["<unk>"])
pad_ix = vocab['<pad>']

#hyper-parameter
input_dim  = len(vocab)
hid_dim    = 256
emb_dim    = 300         
output_dim = 5
num_layers = 2
bidirectional = True
dropout = 0.5

model = LSTM(input_dim, emb_dim, hid_dim, output_dim, num_layers, bidirectional, dropout, pad_ix).to(device)
save_path = f'models/LSTM_TreeBank.pt'
model.load_state_dict(torch.load(save_path))

def prediction(test_str_list):
    result = list()
    for test_str in test_str_list:
        text = torch.tensor(text_pipeline(test_str)).to(device).reshape(1, -1)
        # text_list = [x.item() for x in text]
        text_length = torch.tensor([text.size(1)]).to(dtype=torch.int64)
        output = model(text, text_length).squeeze(1)
        predicted = torch.max(output.data, 1)[1].detach().cpu().numpy()[0]
        result.append((test_str, predicted))
    return result

from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
nlp = spacy.load('en_core_web_md')

def Reddit(name,reddit,limit):
    subreddit = reddit.subreddit(name)
    topics = [*subreddit.top(limit=limit)] # top posts all time
    # print(len(topics))
    df_topics = pd.DataFrame({"title": [n.title for n in topics]})
    df_result = prediction(df_topics['title'])
    return df_result

def PosNeg(result):
    df = pd.DataFrame(result)
    df = df.rename(columns={0: 'Title', 1: 'Rating'})
    print(df['Rating'].value_counts())
    #POS/NEG
    df['clean'] = df['Title'].apply(preprocessing)
    df_pos = df[df['Rating'].isin([3,4])] #postive
    df_neg = df[df['Rating'].isin([0,1])] #negative
    common_words_pos = findvocab(df_pos['clean'])
    common_words_neg = findvocab(df_neg['clean'])
    #Graph
    score = df['Rating'].value_counts().sort_index()
    score = score.rename(index={0: "Very Negative", 1: "Negative", 2: "Normal",3: "Positive",4:"Very Positive"})
    score.plot.barh()
    plt.title('Sentiment classification')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    output = io.BytesIO() #Retrieve the entire contents of the BytesIO object
    plt.savefig(output, format='png')
    plot_url = base64.b64encode(output.getvalue()).decode('utf-8')
    return common_words_pos,common_words_neg,plot_url

def preprocessing(sentence):
    stopwords = list(STOP_WORDS)
    doc = nlp(sentence)
    cleaned_tokens = []

    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and \
                token.pos_ != 'SYM':
            cleaned_tokens.append(token.lemma_.lower().strip())

    return " ".join(cleaned_tokens)

def findvocab(corpus):
    #data tokenized
    corpus_tokenized = [sent.split(" ") for sent in corpus]
    #2. numericalize (vocab)
    #2.1 get all the unique words
    #we want to flatten unit (basically merge all list)
    flatten = lambda l: [item for sublist in l for item in sublist]
    vocabs = list(flatten(corpus_tokenized))
    voc_size = len(vocabs)
    print('Vocab Size :',voc_size)
    word_freq = Counter(vocabs)
    common_words = word_freq.most_common(3)
    return common_words

