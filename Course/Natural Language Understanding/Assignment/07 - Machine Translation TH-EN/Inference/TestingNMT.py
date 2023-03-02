import torch
import pickle
from NMTAttention_model import *
from attacut import tokenize, Tokenizer
from torchtext.data.utils import get_tokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SRC_LANGUAGE,TRG_LANGUAGE = 'th', 'en'

token_transform = {}
token_transform[SRC_LANGUAGE] = Tokenizer(model="attacut-sc")
token_transform[TRG_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

# Load data (vocab and transform)
import pickle

with open('vocab_transform.pickle', 'rb') as handle:
    vocab_transform = pickle.load(handle)
print(vocab_transform)

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    global func
    def func(txt_input):
        for transform in transforms:
            if transform == token_transform[SRC_LANGUAGE]:
                txt_input = transform.tokenize(txt_input)
            else:
                txt_input = transform(txt_input)
        return txt_input

    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

# src and trg language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

def initialize_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

def translation(source, variants, save_path, device):
    src_text = text_transform[SRC_LANGUAGE](source).to(device)
    target = "It is fake target"*20
    trg_text = text_transform[TRG_LANGUAGE](target).to(device)
    trg_text = trg_text.reshape(-1, 1)
    src_text = src_text.reshape(-1, 1)  #because batch_size is 1
    print('src_text and trg_text shape',src_text.shape, trg_text.shape)
    text_length = torch.tensor([src_text.size(0)]).to(dtype=torch.int64)

    input_dim   = len(vocab_transform[SRC_LANGUAGE])
    output_dim  = len(vocab_transform[TRG_LANGUAGE])
    emb_dim     = 256  
    hid_dim     = 512  
    dropout     = 0.5
    SRC_PAD_IDX = PAD_IDX

    attn = Attention(hid_dim, variants=variants)
    enc  = Encoder(input_dim,  emb_dim,  hid_dim, dropout)
    dec  = Decoder(output_dim, emb_dim,  hid_dim, dropout, attn)

    model = Seq2SeqPackedAttention(enc, dec, SRC_PAD_IDX, device).to(device)
    model.apply(initialize_weights)
    # print(model)

    model.load_state_dict(torch.load(save_path))
    model.eval()

    with torch.no_grad():
        output, attentions = model(src_text, text_length, trg_text, 0) #turn off teacher forcing
    #trg_len, batch_size, trg_output_dim
    output = output.squeeze(1)
    #trg_len, trg_output_dim
    output = output[1:]
    output_max = output.argmax(1) #returns max indices
    mapping = vocab_transform[TRG_LANGUAGE].get_itos()

    predict_setence = []
    for token in output_max:
        if mapping[token.item()] == '<eos>':
            return ' '.join(predict_setence)
        
        predict_setence.append(mapping[token.item()])

    return ' '.join(predict_setence)