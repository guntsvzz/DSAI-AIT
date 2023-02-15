from LSTM_model import *
import torch
import pytreebank
import spacy
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

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